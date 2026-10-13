"""E4 (stretch) - Isigidimi vs Imvo attribution in the 1884-1888 overlap window.

The one MODEL COMPARISON in the project, so it follows the brief's rules:
edition-level train/validation/test split, hyper-parameters tuned on validation,
a single final evaluation on test.

Inputs compared: raw historical spelling vs normalised spelling.
Models: TF-IDF + logistic regression (word 1-2 grams or char 2-5 grams, C tuned).
        Fine-tuned AfroXLMR is left as a TODO (only if E1-E3 finish by 20 Oct).

Usage:
    python -m nlp2.e4_attribution

Confound check: digits are removed and the top features are saved, so we can see
whether the classifier learned language/themes or page artefacts.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config
from .normalise import NormParams, Normaliser, load_lexicon
from .utils import bootstrap_ci, get_logger, set_seed

log = get_logger("e4")
C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_overlap(normalised: bool) -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DIR / "segments_xh.csv.gz")
    lo, hi = config.OVERLAP_YEARS
    df = df[df["year"].between(lo, hi)].copy()
    df["text"] = df["text"].map(lambda t: re.sub(r"\d+", " ", t))
    if normalised:
        params = NormParams()
        if config.NORMALISER_PARAMS_JSON.exists():
            params = NormParams(**json.loads(config.NORMALISER_PARAMS_JSON.read_text()))
        df["text"] = df["text"].map(Normaliser(load_lexicon(), params).text)
    # Balance the two papers within each split (Imvo is much larger).
    parts = []
    for _, g in df.groupby("split"):
        n = g["paper"].value_counts().min()
        parts += [gp.sample(n=n, random_state=config.SEED) for _, gp in g.groupby("paper")]
    return pd.concat(parts).reset_index(drop=True)


def vectoriser(kind: str):
    from sklearn.feature_extraction.text import TfidfVectorizer

    if kind == "word_1_2":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)


def run_tfidf(df: pd.DataFrame, input_name: str) -> tuple[dict, list[dict], pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    tr, va, te = (df[df["split"] == s] for s in ("train", "validation", "test"))
    grid_rows, best = [], None
    for feats in ("word_1_2", "char_2_5"):
        vec = vectoriser(feats).fit(tr["text"])
        Xtr, Xva = vec.transform(tr["text"]), vec.transform(va["text"])
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=config.SEED).fit(Xtr, tr["paper"])
            f1 = f1_score(va["paper"], clf.predict(Xva), average="macro")
            grid_rows.append({"input": input_name, "features": feats, "C": C, "val_macro_f1": f1})
            if best is None or f1 > best[0]:
                best = (f1, feats, C, vec, clf)
    _, feats, C, vec, clf = best
    pred = clf.predict(vec.transform(te["text"]))
    correct = (pred == te["paper"].to_numpy()).astype(float)
    acc, lo, hi = bootstrap_ci(correct)
    result = {
        "input": input_name, "model": "tfidf_logreg", "features": feats, "C": C,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "test_macro_f1": f1_score(te["paper"], pred, average="macro"),
        "test_accuracy": accuracy_score(te["paper"], pred), "acc_lo": lo, "acc_hi": hi,
    }
    names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    top = pd.DataFrame(
        [{"input": input_name, "paper": clf.classes_[0], "feature": names[i], "weight": coefs[i]} for i in order[:25]]
        + [{"input": input_name, "paper": clf.classes_[1], "feature": names[i], "weight": coefs[i]} for i in order[::-1][:25]]
    )
    return result, grid_rows, top


def finetune_transformer(df: pd.DataFrame, input_name: str) -> dict:
    """TODO (stretch): fine-tune config.HF_MODELS['afroxlmr'] for sequence classification.

    Tune learning rate {1e-5, 3e-5, 5e-5} and epochs {2, 3} on validation, report test
    macro-F1 with the same splits as the TF-IDF model. Only start if E1-E3 are done.
    """
    raise NotImplementedError


def main() -> None:
    set_seed()
    results, grids, tops = [], [], []
    for input_name, normalised in (("raw", False), ("normalised", True)):
        df = load_overlap(normalised)
        log.info("%s: %s", input_name, df.groupby(["split", "paper"]).size().to_dict())
        res, grid, top = run_tfidf(df, input_name)
        results.append(res), grids.extend(grid), tops.append(top)
        log.info("%s", res)
    config.ensure_dirs()
    pd.DataFrame(results).to_csv(config.RESULTS_DIR / "e4_attribution.csv", index=False)
    pd.DataFrame(grids).to_csv(config.RESULTS_DIR / "e4_grid.csv", index=False)
    pd.concat(tops).to_csv(config.RESULTS_DIR / "e4_top_features.csv", index=False)


if __name__ == "__main__":
    main()
