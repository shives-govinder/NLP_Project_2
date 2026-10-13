"""E2 - Masked-LM pseudo-perplexity (Salazar et al., 2020) per condition.

Each non-special token is masked in turn and scored; a segment's pseudo-log-likelihood
(PLL) is the sum of those log-probabilities.

Reported per (model, condition, subset):
  pppl_token  exp(-sum PLL / sum tokens)  - compare ACROSS CONDITIONS WITHIN A MODEL only
  bpc         bits per non-space character - PRIMARY cross-condition view: independent of the
              tokeniser and of word division ("e Qonce" vs "eQonce")
  pppl_word   exp(-sum PLL / sum words)   - reported, but word division differs between spellings
  ratio_to_modern  pppl_token / pppl_token of the modern condition in the same language
95% CIs come from a paired bootstrap over segments.

Usage (GPU strongly recommended; e.g. Google Colab):
    python -m nlp2.e2_pseudo_perplexity                       # all models x conditions
    python -m nlp2.e2_pseudo_perplexity --limit 50            # quick smoke test
    python -m nlp2.e2_pseudo_perplexity --models afroxlmr --conditions hist_xh_raw
Per-segment scores are cached in results/e2/, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from . import config
from .utils import available_conditions, bootstrap_ratio_ci, get_logger, load_condition, pick_device, set_seed, subsets

log = get_logger("e2")
CACHE_DIR = config.RESULTS_DIR / "e2"


def segment_pll(text: str, model, tok, device: str, max_tokens: int, batch_size: int) -> dict:
    import torch

    enc = tok(text, truncation=True, max_length=max_tokens, return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0]
    special = set(tok.all_special_ids)
    positions = [i for i in range(len(ids)) if int(ids[i]) not in special]
    used_chars = int(offsets[positions[-1]][1]) if positions else 0
    n_words_used = len(text[:used_chars].split())
    n_chars_used = len("".join(text[:used_chars].split()))

    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            batch = ids.repeat(len(chunk), 1)
            rows = torch.arange(len(chunk))
            batch[rows, chunk] = tok.mask_token_id
            logits = model(input_ids=batch.to(device)).logits
            logp = torch.log_softmax(logits[rows, chunk].float(), dim=-1)
            total += float(logp[rows, ids[chunk].to(device)].sum())
    return {
        "pll": total,
        "n_tokens": len(positions),
        "n_words": max(n_words_used, 1),
        "n_chars": max(n_chars_used, 1),
        "truncated": used_chars < len(text.rstrip()),
    }


def score_condition(model_name: str, cond: str, model, tok, device: str, limit: int | None, overwrite: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{model_name}__{cond}.csv"
    df = load_condition(cond)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=config.SEED)
    done = pd.read_csv(path) if path.exists() and not overwrite else pd.DataFrame(columns=["seg_id"])
    todo = df[~df["seg_id"].isin(done["seg_id"])]
    log.info("%s / %s: %d cached, %d to score", model_name, cond, len(done), len(todo))

    new_rows = []
    for i, row in enumerate(todo.itertuples(index=False), 1):
        new_rows.append({"seg_id": row.seg_id, **segment_pll(row.text, model, tok, device, config.PPPL_MAX_TOKENS, config.PPPL_BATCH_SIZE)})
        if i % 100 == 0 or i == len(todo):  # checkpoint so Colab disconnects lose little
            pd.concat([done, pd.DataFrame(new_rows)]).to_csv(path, index=False)
            log.info("  %d/%d", i, len(todo))
    scored = pd.read_csv(path) if path.exists() else done
    keep = ["seg_id"] + [c for c in ("had_illegible", "clean") if c in df.columns]
    return scored.merge(df[keep], on="seg_id", how="inner")


def summarise(scores: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (model_name, cond), df in scores.items():
        for subset, sdf in subsets(df):
            nll = -sdf["pll"].to_numpy()
            tok_nll, tok_lo, tok_hi = bootstrap_ratio_ci(nll, sdf["n_tokens"].to_numpy())
            word_nll, word_lo, word_hi = bootstrap_ratio_ci(nll, sdf["n_words"].to_numpy())
            char_nll, char_lo, char_hi = bootstrap_ratio_ci(nll, sdf["n_chars"].to_numpy())
            rows.append(
                {
                    "model": model_name, "condition": cond, "subset": subset, "n_segments": len(sdf),
                    "pppl_token": math.exp(tok_nll), "pppl_token_lo": math.exp(tok_lo), "pppl_token_hi": math.exp(tok_hi),
                    "pppl_word": math.exp(word_nll), "pppl_word_lo": math.exp(word_lo), "pppl_word_hi": math.exp(word_hi),
                    "bpc": char_nll / math.log(2), "bpc_lo": char_lo / math.log(2), "bpc_hi": char_hi / math.log(2),
                    "pct_truncated": 100 * sdf["truncated"].astype(bool).mean(),
                }
            )
    res = pd.DataFrame(rows)
    # Ratio to the modern condition of the same language, same model (subset 'all').
    base = res[(res["subset"] == "all") & res["condition"].str.startswith("modern_")]
    base = {(r.model, config.CONDITION_LANG[r.condition]): r.pppl_token for r in base.itertuples()}
    res["ratio_to_modern"] = [
        p / base.get((m, config.CONDITION_LANG[c]), np.nan) for m, c, p in zip(res["model"], res["condition"], res["pppl_token"])
    ]
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(config.HF_MODELS))
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only N segments per condition")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    set_seed()
    device = pick_device()
    log.info("Device: %s", device)
    conditions = args.conditions or available_conditions()

    scores = {}
    for model_name in args.models:
        repo = config.HF_MODELS[model_name]
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForMaskedLM.from_pretrained(repo).to(device).eval()
        for cond in conditions:
            scores[(model_name, cond)] = score_condition(model_name, cond, model, tok, device, args.limit, args.overwrite)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    res = summarise(scores)
    res.to_csv(config.RESULTS_DIR / "e2_pseudo_perplexity.csv", index=False)
    log.info("\n%s", res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
