"""Figures and tables for the extended abstract, built only from results/*.csv.

Usage:
    python -m nlp2.plots

Outputs:
    figures/fig1_tokeniser_and_pppl.png   (a) E1 tokens per 100 characters, (b) E2 bits per character
    results/table1.tex                     E3 LID recall + normaliser test accuracy
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .utils import get_logger

log = get_logger("plots")
ORDER = list(config.CONDITIONS)
SHORT = {"modern_xh": "Mod. xh", "hist_xh_raw": "Hist. xh", "hist_xh_norm": "Hist. xh\n(norm.)", "modern_en": "Mod. en", "hist_en": "Hist. en"}
PALETTE = ["#2a78d6", "#7d7b73", "#b4b2a9", "#d9d7cf"]


def grouped_bars(ax, df: pd.DataFrame, group_col: str, value: str, lo: str | None, hi: str | None, ylabel: str) -> None:
    groups = list(df[group_col].unique())
    conds = [c for c in ORDER if c in set(df["condition"])]
    width = 0.8 / max(len(groups), 1)
    for gi, g in enumerate(groups):
        sub = df[df[group_col] == g].set_index("condition").reindex(conds)
        x = np.arange(len(conds)) + (gi - (len(groups) - 1) / 2) * width
        yerr = None
        if lo and hi:
            yerr = np.vstack([sub[value] - sub[lo], sub[hi] - sub[value]])
        ax.bar(x, sub[value], width=width, color=PALETTE[gi % len(PALETTE)], label=g, yerr=yerr, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(range(len(conds)), [SHORT[c] for c in conds], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6)


def fig1() -> None:
    e1p, e2p = config.RESULTS_DIR / "e1_tokeniser_cost.csv", config.RESULTS_DIR / "e2_pseudo_perplexity.csv"
    if not (e1p.exists() and e2p.exists()):
        log.warning("Need %s and %s for Figure 1", e1p.name, e2p.name)
        return
    e1 = pd.read_csv(e1p).query("subset == 'all'")
    e2 = pd.read_csv(e2p).query("subset == 'all'")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.4), dpi=300)
    grouped_bars(a, e1, "tokeniser", "tokens_per_100c", "tokens_per_100c_lo", "tokens_per_100c_hi", "Subword tokens per 100 characters")
    a.set_title("(a) Tokeniser cost", fontsize=8)
    grouped_bars(b, e2, "model", "bpc", "bpc_lo", "bpc_hi", "Bits per character (lower = better)")
    b.set_title("(b) Masked-LM pseudo-log-likelihood", fontsize=8)
    fig.tight_layout()
    path = config.FIGURES_DIR / "fig1_tokeniser_and_pppl.png"
    fig.savefig(path)
    log.info("Saved %s", path)


def table1() -> None:
    e3p, normp = config.RESULTS_DIR / "e3_language_id.csv", config.RESULTS_DIR / "normaliser_test.csv"
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Condition & LID recall (\%) & 95\% CI \\", r"\midrule"]
    if e3p.exists():
        e3 = pd.read_csv(e3p).query("subset == 'all'").set_index("condition")
        for c in [c for c in ORDER if c in e3.index]:
            r = e3.loc[c]
            lines.append(f"{config.CONDITIONS[c]} & {100*r.recall:.1f} & [{100*r.recall_lo:.1f}, {100*r.recall_hi:.1f}] \\\\")
    if normp.exists():
        lines += [r"\midrule", r"Normaliser (gold test) & Accuracy (\%) & Rewrite prec. (\%) \\", r"\midrule"]
        for r in pd.read_csv(normp).itertuples():
            lines.append(f"{r.system.replace('_', ' ')} & {100*r.accuracy:.1f} & {100*r.rewrite_precision:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = config.RESULTS_DIR / "table1.tex"
    path.write_text("\n".join(lines) + "\n")
    log.info("Saved %s", path)


def main() -> None:
    config.ensure_dirs()
    fig1()
    table1()


if __name__ == "__main__":
    main()
