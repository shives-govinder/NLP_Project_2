"""Step 3: lexicon-gated aspirate-h restoration (historical -> modern spelling).

Idea: 19th-century isiXhosa usually wrote aspirated consonants without "h"
(e.g. 'ukuti' for modern 'ukuthi'). For each historical word we generate the
candidates obtained by inserting "h" after aspirable consonants and accept a
candidate only if it is attested in the modern lexicon (MasakhaNEWS train+val)
often enough. Words with no attested candidate are left untouched, so the
normaliser is conservative by construction.

Usage:
    python -m nlp2.normalise candidates   # write words for the gold set to annotate
    python -m nlp2.normalise tune         # grid-search on gold 'validation', report on 'test'
    python -m nlp2.normalise apply        # write conditions/hist_xh_norm.csv

Gold set (annotations/gold_normalisation.csv) columns:
    historical, modern, split (validation|test), annotator, notes
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config
from .data_prep import WORD_RE
from .utils import get_logger, load_condition, save_condition, set_seed

log = get_logger("normalise")


@dataclass(frozen=True)
class NormParams:
    min_freq: int = 2
    ratio: float = 2.0
    max_insert: int = 2


def load_lexicon() -> dict[str, int]:
    if not config.LEXICON_CSV.exists():
        raise FileNotFoundError(f"{config.LEXICON_CSV} missing - run `python -m nlp2.modern_data` first.")
    df = pd.read_csv(config.LEXICON_CSV, keep_default_na=False)
    return dict(zip(df["word"].astype(str), df["freq"].astype(int)))


def h_candidates(word: str, max_insert: int) -> list[str]:
    """All spellings with 1..max_insert 'h' inserted after aspirable consonants.

    'ukuti' -> ['ukhuti', 'ukuthi', 'ukhuthi'] (max_insert=2)
    """
    w = word.lower()
    slots = [i for i, ch in enumerate(w) if ch in config.ASPIRABLE and (i + 1 >= len(w) or w[i + 1] != "h")]
    out = []
    for k in range(1, min(max_insert, len(slots)) + 1):
        for combo in itertools.combinations(slots, k):
            chars = list(w)
            for pos in reversed(combo):
                chars.insert(pos + 1, "h")
            out.append("".join(chars))
    return out


def _match_case(src: str, tgt: str) -> str:
    if src.isupper() and len(src) > 1:
        return tgt.upper()
    if src[:1].isupper():
        return tgt[:1].upper() + tgt[1:]
    return tgt


class Normaliser:
    def __init__(self, lexicon: dict[str, int], params: NormParams = NormParams()):
        self.lex = lexicon
        self.p = params
        self._cache: dict[str, str] = {}

    def word(self, w: str) -> str:
        key = w.lower()
        if key not in self._cache:
            self._cache[key] = self._best(key)
        return _match_case(w, self._cache[key])

    def _best(self, w: str) -> str:
        orig_freq = self.lex.get(w, 0)
        best, best_freq = w, orig_freq
        for cand in h_candidates(w, self.p.max_insert):
            f = self.lex.get(cand, 0)
            if f >= self.p.min_freq and f >= self.p.ratio * max(orig_freq, 1) and f > best_freq:
                best, best_freq = cand, f
        return best

    def text(self, t: str) -> str:
        return WORD_RE.sub(lambda m: self.word(m.group(0)), t)


def naive_all(word: str) -> str:
    """Baseline: insert h after EVERY aspirable consonant (no lexicon)."""
    w = word.lower()
    return "".join(ch + ("h" if ch in config.ASPIRABLE and (i + 1 >= len(w) or w[i + 1] != "h") else "") for i, ch in enumerate(w))


def evaluate(pred: list[str], gold_hist: list[str], gold_mod: list[str]) -> dict:
    """Word-level metrics on the gold set (all lower-cased)."""
    pred = [p.lower() for p in pred]
    hist = [h.lower() for h in gold_hist]
    mod = [m.lower() for m in gold_mod]
    correct = np.array([p == m for p, m in zip(pred, mod)])
    changed = np.array([p != h for p, h in zip(pred, hist)])
    needed = np.array([m != h for m, h in zip(mod, hist)])
    return {
        "n": len(pred),
        "accuracy": float(correct.mean()),
        "rewrite_precision": float((changed & correct).sum() / max(changed.sum(), 1)),
        "rewrite_recall": float((needed & correct & changed).sum() / max(needed.sum(), 1)),
        "change_rate": float(changed.mean()),
        "needed_rate": float(needed.mean()),
    }


def load_gold() -> pd.DataFrame:
    gold = pd.read_csv(config.GOLD_CSV, keep_default_na=False)
    gold = gold[(gold["historical"] != "") & (gold["modern"] != "")]
    if gold.empty:
        raise ValueError(f"{config.GOLD_CSV} has no annotated rows yet.")
    return gold


def cmd_candidates(n: int) -> None:
    """Sample word types across frequency bands, with a context sentence, for annotation.

    No suggestion is pre-filled so the annotator is not anchored by our normaliser.
    """
    df = load_condition("hist_xh_raw")
    counts = pd.Series([w.lower() for t in df["text"] for w in WORD_RE.findall(t)]).value_counts()
    counts = counts[counts.index.str.len() >= 3]
    bands = pd.qcut(counts.rank(method="first"), q=3, labels=["rare", "mid", "frequent"])
    per_band = n // 3
    picked = pd.concat(
        [counts[bands == b].sample(n=min(per_band, int((bands == b).sum())), random_state=config.SEED) for b in bands.cat.categories]
    )
    rows = []
    for word in picked.index:
        ctx = next((t for t in df["text"] if re.search(rf"\b{re.escape(word)}\b", t, re.I)), "")
        rows.append({"historical": word, "modern": "", "freq_in_sample": int(picked[word]), "context": ctx[:200]})
    out = pd.DataFrame(rows).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    out["split"] = np.where(np.arange(len(out)) % 2 == 0, "validation", "test")
    out["annotator"], out["notes"] = "", ""
    path = config.ANNOTATIONS_DIR / "gold_candidates_to_annotate.csv"
    out.to_csv(path, index=False)
    log.info("Wrote %d words to %s. Fill in 'modern' (same as historical if unchanged) and save as %s",
             len(out), path, config.GOLD_CSV.name)


def cmd_tune() -> None:
    lex = load_lexicon()
    gold = load_gold()
    val, test = gold[gold["split"] == "validation"], gold[gold["split"] == "test"]
    grid = [NormParams(*vals) for vals in itertools.product(*config.NORMALISER_GRID.values())]

    rows = []
    for p in grid:
        norm = Normaliser(lex, p)
        m = evaluate([norm.word(w) for w in val["historical"]], list(val["historical"]), list(val["modern"]))
        rows.append({**asdict(p), **{f"val_{k}": v for k, v in m.items()}})
    grid_df = pd.DataFrame(rows).sort_values(["val_accuracy", "val_rewrite_precision"], ascending=False)
    grid_df.to_csv(config.RESULTS_DIR / "normaliser_grid.csv", index=False)
    best = NormParams(**{k: grid_df.iloc[0][k] for k in ("min_freq", "ratio", "max_insert")})
    best = NormParams(int(best.min_freq), float(best.ratio), int(best.max_insert))
    log.info("Best on validation: %s", best)

    systems = {
        "identity": lambda w: w,
        "naive_all_h": naive_all,
        "lexicon_gated": Normaliser(lex, best).word,
    }
    report = []
    for name, fn in systems.items():
        m = evaluate([fn(w) for w in test["historical"]], list(test["historical"]), list(test["modern"]))
        report.append({"system": name, **m})
    pd.DataFrame(report).to_csv(config.RESULTS_DIR / "normaliser_test.csv", index=False)
    config.NORMALISER_PARAMS_JSON.write_text(json.dumps(asdict(best), indent=2))
    log.info("Test results:\n%s", pd.DataFrame(report).to_string(index=False))


def cmd_apply() -> None:
    lex = load_lexicon()
    params = NormParams()
    if config.NORMALISER_PARAMS_JSON.exists():
        params = NormParams(**json.loads(config.NORMALISER_PARAMS_JSON.read_text()))
    else:
        log.warning("No tuned params found; using defaults %s. Run `tune` first for the real experiment.", params)
    norm = Normaliser(lex, params)
    raw = load_condition("hist_xh_raw")
    out = raw.copy()
    out["text"] = raw["text"].map(norm.text)
    changed = sum(a != b for t1, t2 in zip(raw["text"], out["text"]) for a, b in zip(WORD_RE.findall(t1), WORD_RE.findall(t2)))
    total = int(raw["n_words"].sum())
    log.info("Changed %d of %d words (%.1f%%)", changed, total, 100 * changed / max(total, 1))
    log.info("Saved %s", save_condition("hist_xh_norm", out))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--n", type=int, default=300)
    sub.add_parser("tune")
    sub.add_parser("apply")
    args = ap.parse_args(argv)
    set_seed()
    config.ensure_dirs()
    {"candidates": lambda: cmd_candidates(args.n), "tune": cmd_tune, "apply": cmd_apply}[args.cmd]()


if __name__ == "__main__":
    main()
