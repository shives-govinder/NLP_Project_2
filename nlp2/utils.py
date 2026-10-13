"""Shared helpers: seeding, logging, condition I/O and bootstrap CIs."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence per-request HTTP logs from Hugging Face downloads.
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)  # unauthenticated-request notice
    return logging.getLogger(name)


def set_seed(seed: int = config.SEED) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device() -> str:
    """cuda > mps (Apple silicon) > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Condition files: one CSV per experimental condition, same schema for all
#   seg_id, text, n_words, + optional metadata (paper, year, edition, split, ...)
# ---------------------------------------------------------------------------
def condition_path(name: str) -> Path:
    if name not in config.CONDITIONS:
        raise KeyError(f"Unknown condition {name!r}; expected one of {list(config.CONDITIONS)}")
    return config.CONDITIONS_DIR / f"{name}.csv"


def save_condition(name: str, df: pd.DataFrame) -> Path:
    required = {"seg_id", "text", "n_words"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Condition {name} is missing columns {missing}")
    config.ensure_dirs()
    path = condition_path(name)
    df.to_csv(path, index=False)
    return path


def load_condition(name: str) -> pd.DataFrame:
    path = condition_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the data steps first (see README.md, steps 1-4)."
        )
    return pd.read_csv(path)


def available_conditions() -> list[str]:
    return [c for c in config.CONDITIONS if condition_path(c).exists()]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = config.BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap (1 - alpha) CI over segments."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.array([stat(arr[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(arr)), float(lo), float(hi)


def bootstrap_ratio_ci(
    num: Sequence[float],
    den: Sequence[float],
    n_boot: int = config.BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int = config.SEED,
) -> tuple[float, float, float]:
    """Ratio of sums (e.g. corpus-level NLL / tokens) with a paired bootstrap CI."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    boots = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def subsets(df: pd.DataFrame):
    """Yield ('all', df) and, for historical conditions, ('clean', low-OCR-noise segments).

    'clean' = no illegible ('??') words and no words that occur only once in the whole
    historical corpus (see data_prep). This control separates OCR noise from spelling.
    """
    yield "all", df
    if "clean" in df.columns:
        flag = df["clean"].astype(str).str.lower().isin(["true", "1"])
        if 0 < flag.sum() < len(df):
            yield "clean", df[flag]
