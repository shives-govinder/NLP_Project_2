"""Exploratory analysis for the Data section and the supplementary material.

Usage:
    python -m nlp2.eda       # needs outputs/processed/segments_*.csv.gz (run data_prep first)

Outputs:
    results/eda_words_per_year.csv      isiXhosa / English words per paper-year (tracker)
    results/eda_orthography_pairs.csv   historical vs modern spelling counts
    results/eda_segments_summary.csv    segments, words and % with illegible words by paper
    figures/eda_words_per_year.png
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .data_prep import WORD_RE
from .utils import get_logger

log = get_logger("eda")

GREY, BLUE = "#b4b2a9", "#2a78d6"


def words_per_year() -> pd.DataFrame:
    df = pd.read_csv(config.TRACKER_CSV, encoding="utf-8-sig")
    df["paper"] = df["Edition"].str[:4]
    df["year"] = df["Edition"].str[5:9].astype(int)
    for c in ("Xho_Wordcount", "Eng_Wordcount"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    out = df.groupby(["paper", "year"]).agg(
        pages=("Edition", "size"), xh_words=("Xho_Wordcount", "sum"), en_words=("Eng_Wordcount", "sum")
    ).reset_index()
    return out


def plot_words_per_year(wpy: pd.DataFrame) -> None:
    years = range(wpy["year"].min(), wpy["year"].max() + 1)
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=200)
    w = 0.4
    for paper, colour, off, label in (("ISIG", GREY, -w / 2, "Isigidimi sama-Xosa"), ("IMVO", BLUE, w / 2, "Imvo Zabantsundu")):
        sub = wpy[wpy["paper"] == paper].set_index("year").reindex(years).fillna(0)
        ax.bar([y + off for y in years], sub["xh_words"] / 1000, width=w, color=colour, label=label)
    ax.axvspan(1883.5, 1888.5, color="#000000", alpha=0.05, lw=0)
    ax.set_ylabel("isiXhosa words (thousands)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = config.FIGURES_DIR / "eda_words_per_year.png"
    fig.savefig(path)
    log.info("Saved %s", path)


def orthography_pairs(segments: pd.DataFrame) -> pd.DataFrame:
    counts = Counter(w.lower() for t in segments["text"] for w in WORD_RE.findall(t))
    return pd.DataFrame(
        [{"historical": h, "hist_count": counts[h], "modern": m, "modern_count": counts[m]} for h, m in config.ORTHOGRAPHY_PAIRS]
    )


def main() -> None:
    config.ensure_dirs()
    wpy = words_per_year()
    wpy.to_csv(config.RESULTS_DIR / "eda_words_per_year.csv", index=False)
    plot_words_per_year(wpy)

    seg_path = config.PROCESSED_DIR / "segments_xh.csv.gz"
    if not seg_path.exists():
        log.warning("%s missing - run `python -m nlp2.data_prep` for the segment-level EDA.", seg_path)
        return
    xh = pd.read_csv(seg_path)
    pairs = orthography_pairs(xh)
    pairs.to_csv(config.RESULTS_DIR / "eda_orthography_pairs.csv", index=False)
    log.info("Orthography pairs:\n%s", pairs.to_string(index=False))

    summary = xh.groupby("paper").agg(
        segments=("seg_id", "size"), words=("n_words", "sum"), pct_with_illegible=("had_illegible", "mean")
    )
    summary["pct_with_illegible"] *= 100
    summary.to_csv(config.RESULTS_DIR / "eda_segments_summary.csv")
    log.info("Segments summary:\n%s", summary.round(2).to_string())


if __name__ == "__main__":
    main()
