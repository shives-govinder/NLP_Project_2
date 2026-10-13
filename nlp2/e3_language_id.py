"""E3 - Off-the-shelf language identification (GlotLID) per condition.

Gold labels come for free: XhTxt segments are isiXhosa, EnTxt segments are English.
LID recall matters because web-scale corpora are filtered by LID: text that is not
recognised as isiXhosa never reaches the training data of future models.

Outputs
  e3_language_id.csv   recall of the expected label per (condition, subset), bootstrap 95% CI,
                       plus `recall_len_std`: recall re-weighted to the modern condition's
                       length distribution (LID is worse on short text and the historical
                       sample has more short sentences, so raw recall mixes the two effects)
  e3_by_length.csv     recall per sentence-length bin
  e3_confusions.csv    labels predicted instead of the expected one
  e3_predictions.csv   one row per segment (for error analysis)

Usage:
    python -m nlp2.e3_language_id
Runs on CPU. Needs: fasttext (or fasttext-wheel) and huggingface_hub (~1.7 GB model download once).
"""

from __future__ import annotations

import math

import pandas as pd

from . import config
from .utils import available_conditions, bootstrap_ci, get_logger, load_condition, set_seed, subsets

log = get_logger("e3")


def load_glotlid():
    import fasttext
    from huggingface_hub import hf_hub_download

    fasttext.FastText.eprint = lambda *args, **kwargs: None  # silence the load_model notice
    path = hf_hub_download(config.GLOTLID_REPO, config.GLOTLID_FILE)
    model = fasttext.load_model(path)
    labels = {l.replace("__label__", "") for l in model.get_labels()}
    for lang, lab in config.LID_EXPECTED.items():
        if lab not in labels:
            raise ValueError(f"GlotLID has no label {lab!r} for {lang}; update config.LID_EXPECTED")
    return model


def predict(model, text: str) -> tuple[str, float]:
    line = " ".join(text.split())  # fastText needs a single line
    try:
        # Low-level call avoids the numpy>=2 'copy=False' bug in fasttext's Python wrapper.
        (prob, label), = model.f.predict(line, 1, 0.0, "strict")
    except Exception:
        labels, probs = model.predict(line, k=1)
        label, prob = labels[0], probs[0]
    return label.replace("__label__", ""), float(prob)


def mcnemar_exact(a_correct: pd.Series, b_correct: pd.Series) -> dict:
    """Paired exact McNemar test on the same segments (two-sided binomial on discordant pairs)."""
    b = int((a_correct & ~b_correct).sum())   # only A right
    c = int((~a_correct & b_correct).sum())   # only B right
    n, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_raw_correct": b, "only_norm_correct": c, "p_value": p}


def length_bin(n_words: pd.Series) -> pd.Series:
    return pd.cut(n_words, config.LENGTH_BINS, labels=config.LENGTH_LABELS)


def main() -> None:
    set_seed()
    model = load_glotlid()
    preds_all = []
    for cond in available_conditions():
        df = load_condition(cond)
        expected = config.LID_EXPECTED[config.CONDITION_LANG[cond]]
        p = [predict(model, t) for t in df["text"]]
        df = df.assign(condition=cond, expected=expected, pred=[x for x, _ in p], conf=[c for _, c in p])
        df["correct"] = df["pred"] == expected
        df["len_bin"] = length_bin(df["n_words"])
        preds_all.append(df)
    preds = pd.concat(preds_all, ignore_index=True)

    # Length weights from the modern condition of the same language.
    weights = {}
    for lang in ("xh", "en"):
        m = preds[preds["condition"] == f"modern_{lang}"]
        if len(m):
            weights[lang] = m["len_bin"].value_counts(normalize=True)

    rows, by_len, confusions = [], [], []
    for cond, df in preds.groupby("condition", sort=False):
        lang = config.CONDITION_LANG[cond]
        for subset, sdf in subsets(df):
            mean, lo, hi = bootstrap_ci(sdf["correct"].astype(float))
            per_bin = sdf.groupby("len_bin", observed=False)["correct"].mean()
            w = weights.get(lang)
            std = float((per_bin * w).sum() / w[per_bin.notna()].sum()) if w is not None else float("nan")
            rows.append({"condition": cond, "subset": subset, "expected": sdf["expected"].iloc[0],
                         "n_segments": len(sdf), "recall": mean, "recall_lo": lo, "recall_hi": hi,
                         "recall_len_std": std, "mean_conf": sdf["conf"].mean(),
                         "mean_words": sdf["n_words"].mean()})
            log.info("%-13s %-5s recall %.3f [%.3f, %.3f]  length-standardised %.3f  (mean %.1f words)",
                     cond, subset, mean, lo, hi, std, sdf["n_words"].mean())
        for b, g in df.groupby("len_bin", observed=True):
            by_len.append({"condition": cond, "len_bin": b, "n": len(g), "recall": g["correct"].mean()})
        wrong = df.loc[~df["correct"], "pred"].value_counts().head(10)
        confusions += [{"condition": cond, "predicted": lab, "count": int(n), "share_of_errors": n / max((~df["correct"]).sum(), 1)}
                       for lab, n in wrong.items()]

    # Raw vs normalised are the SAME segments, so compare them with a paired test.
    raw = preds[preds["condition"] == "hist_xh_raw"].set_index("seg_id")["correct"]
    norm = preds[preds["condition"] == "hist_xh_norm"].set_index("seg_id")["correct"]
    if len(raw) and len(norm):
        both = raw.index.intersection(norm.index)
        test = mcnemar_exact(raw.loc[both], norm.loc[both])
        pd.DataFrame([test]).to_csv(config.RESULTS_DIR / "e3_mcnemar_raw_vs_norm.csv", index=False)
        log.info("Raw vs normalised (paired McNemar): %s", test)

    config.ensure_dirs()
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "e3_language_id.csv", index=False)
    pd.DataFrame(by_len).to_csv(config.RESULTS_DIR / "e3_by_length.csv", index=False)
    pd.DataFrame(confusions).to_csv(config.RESULTS_DIR / "e3_confusions.csv", index=False)
    keep = ["seg_id", "condition", "expected", "pred", "conf", "correct", "n_words", "len_bin", "text"]
    preds[keep].to_csv(config.RESULTS_DIR / "e3_predictions.csv", index=False)
    log.info("By length:\n%s", pd.DataFrame(by_len).pivot(index="condition", columns="len_bin", values="recall").reindex(columns=config.LENGTH_LABELS).round(3).to_string())


if __name__ == "__main__":
    main()
