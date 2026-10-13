"""E1 - Tokeniser cost ("tokenisation tax") per condition.

Primary metric
  tokens_per_100c  subword tokens per 100 non-space characters (bootstrap 95% CI).
                   Comparable across spellings AND across word-division conventions.

Secondary metrics
  fertility        subword tokens per whitespace word. NOT comparable between historical and
                   modern isiXhosa: 19th-century text writes prefixes as separate words
                   ("e Qonce" vs modern "eQonce"), so its words are ~15% shorter.
  chars_per_word   mean non-space characters per whitespace word (documents that confound)
  pct_words_3plus  share of word tokens split into >= 3 subword pieces
  unk_rate         share of subword tokens that are <unk> (HF tokenisers only)

Usage:
    python -m nlp2.e1_tokeniser_cost
Runs on CPU in about a minute. Needs: transformers, sentencepiece, tiktoken.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import pandas as pd

from . import config
from .utils import available_conditions, bootstrap_ci, get_logger, load_condition, set_seed, subsets

log = get_logger("e1")


def _vocab_signature(tok) -> tuple:
    vocab = tok.get_vocab()
    return (len(vocab), hash(frozenset(vocab.items())))


def load_tokenisers() -> dict[str, dict]:
    """name -> {'encode': text -> list, 'unk': token or None}. Duplicate vocabularies are dropped."""
    from transformers import AutoTokenizer

    toks: dict[str, dict] = {}
    seen: dict[tuple, str] = {}
    specs = {name: (repo, False) for name, repo in config.HF_MODELS.items()}
    specs.update(config.E1_EXTRA_TOKENISERS)
    for name, (repo, remote) in specs.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=remote)
        except Exception as exc:  # gated / offline / missing remote code
            log.warning("Skipping %s (%s): %s", name, repo, str(exc).splitlines()[0][:160])
            continue
        sig = _vocab_signature(tok)
        if sig in seen:
            log.warning("%s has the same vocabulary as %s - reported once as %s", name, seen[sig], seen[sig])
            continue
        seen[sig] = name
        toks[name] = {"encode": tok.tokenize, "unk": tok.unk_token}
    try:
        import tiktoken

        enc = tiktoken.get_encoding(config.TIKTOKEN_ENCODING)
        toks[f"tiktoken_{config.TIKTOKEN_ENCODING}"] = {"encode": enc.encode, "unk": None}
    except ImportError:
        log.warning("tiktoken not installed - skipping the GPT-style tokeniser")
    log.info("Tokenisers: %s", list(toks))
    return toks


def word_split_rate(texts: pd.Series, encode: Callable) -> float:
    """Token-frequency-weighted share of words split into >= 3 pieces."""
    counts = Counter(w for t in texts for w in t.split())
    pieces = {w: len(encode(" " + w)) for w in counts}
    total = sum(counts.values())
    return sum(c for w, c in counts.items() if pieces[w] >= 3) / max(total, 1)


def run() -> pd.DataFrame:
    set_seed()
    toks = load_tokenisers()
    rows, per_seg = [], []
    for cond in available_conditions():
        df = load_condition(cond)
        for subset, sdf in subsets(df):
            n_words = sdf["text"].str.split().str.len().clip(lower=1).to_numpy()
            n_chars = sdf["text"].str.replace(r"\s+", "", regex=True).str.len().clip(lower=1).to_numpy()
            for tname, t in toks.items():
                n_tok = [len(t["encode"](txt)) for txt in sdf["text"]]
                per100 = [100 * k / c for k, c in zip(n_tok, n_chars)]
                fert = [k / w for k, w in zip(n_tok, n_words)]
                p_mean, p_lo, p_hi = bootstrap_ci(per100)
                f_mean, f_lo, f_hi = bootstrap_ci(fert)
                unk = float("nan")
                if t["unk"]:
                    toks_all = [tok for txt in sdf["text"] for tok in t["encode"](txt)]
                    unk = sum(tok == t["unk"] for tok in toks_all) / max(len(toks_all), 1)
                rows.append(
                    {
                        "tokeniser": tname, "condition": cond, "subset": subset, "n_segments": len(sdf),
                        "tokens_per_100c": p_mean, "tokens_per_100c_lo": p_lo, "tokens_per_100c_hi": p_hi,
                        "fertility": f_mean, "fertility_lo": f_lo, "fertility_hi": f_hi,
                        "chars_per_word": n_chars.sum() / n_words.sum(),
                        "pct_words_3plus": 100 * word_split_rate(sdf["text"], t["encode"]),
                        "unk_rate": unk,
                    }
                )
                if subset == "all":
                    per_seg += [
                        {"tokeniser": tname, "condition": cond, "seg_id": s, "tokens": k, "chars": c, "words": w}
                        for s, k, c, w in zip(sdf["seg_id"], n_tok, n_chars, n_words)
                    ]
                log.info("%-20s %-13s %-5s tokens/100c %5.1f [%4.1f, %4.1f]   tokens/word %.2f",
                         tname, cond, subset, p_mean, p_lo, p_hi, f_mean)
    res = pd.DataFrame(rows)
    config.ensure_dirs()
    res.to_csv(config.RESULTS_DIR / "e1_tokeniser_cost.csv", index=False)
    pd.DataFrame(per_seg).to_csv(config.RESULTS_DIR / "e1_per_segment.csv.gz", index=False)
    return res


if __name__ == "__main__":
    run()
