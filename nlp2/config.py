"""Central configuration: paths, seeds and experiment settings.

Every script reads from here, so changing a path or a sample size happens in one
place. Paths can be overridden with environment variables (see README.md).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# By default the raw data sits next to this code folder:
#   NLP Project 2/
#     data/TxtLangFiles1870-1894_TB/{XhTxt,EnTxt,LanguageSplitTracker70-94.csv}
#     NLP_Assignment_2_Code/   <- this repo
DATA_DIR = Path(os.environ.get("NLP2_DATA_DIR", ROOT.parent / "data"))
LANG_SPLIT_DIR = DATA_DIR / "TxtLangFiles1870-1894_TB"
XH_DIR = LANG_SPLIT_DIR / "XhTxt"
EN_DIR = LANG_SPLIT_DIR / "EnTxt"
TRACKER_CSV = LANG_SPLIT_DIR / "LanguageSplitTracker70-94.csv"

OUT_DIR = Path(os.environ.get("NLP2_OUT_DIR", ROOT / "outputs"))
PROCESSED_DIR = OUT_DIR / "processed"
CONDITIONS_DIR = PROCESSED_DIR / "conditions"
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"
ANNOTATIONS_DIR = ROOT / "annotations"
GOLD_CSV = ANNOTATIONS_DIR / "gold_normalisation.csv"
LEXICON_CSV = PROCESSED_DIR / "modern_xh_lexicon.csv"
NORMALISER_PARAMS_JSON = RESULTS_DIR / "normaliser_best_params.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
MIN_WORDS = 5            # drop segments shorter than this
MAX_WORDS = 80           # longer sentences are chunked to this many words
MAX_NONALPHA_RATIO = 0.30  # drop segments whose characters are >30% non-alphabetic
SAMPLE_PER_CONDITION = 1000
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / validation / test, by EDITION
# 'clean' subset (OCR-noise control): no illegible words and at most this share of words that occur
# only once in the whole corpus. A rate (not 'zero') keeps clean and noisy segments similar in length.
CLEAN_MAX_NOISE = 0.10
LENGTH_BINS = [4, 9, 14, 19, 10_000]           # words: 5-9, 10-14, 15-19, 20+
LENGTH_LABELS = ["5-9", "10-14", "15-19", "20+"]
ILLEGIBLE_TOKEN = "<ILL>"  # sentinel used while cleaning, stripped afterwards

# The five experimental conditions. Raw vs normalised historical isiXhosa contain
# the same segments, so spelling is the only thing that differs between them.
CONDITIONS = {
    "modern_xh": "Modern isiXhosa (MasakhaNEWS)",
    "hist_xh_raw": "Historical isiXhosa (raw)",
    "hist_xh_norm": "Historical isiXhosa (normalised)",
    "modern_en": "Modern English (MasakhaNEWS)",
    "hist_en": "Historical English (Imvo)",
}
CONDITION_LANG = {
    "modern_xh": "xh",
    "hist_xh_raw": "xh",
    "hist_xh_norm": "xh",
    "modern_en": "en",
    "hist_en": "en",
}

# ---------------------------------------------------------------------------
# Modern reference corpus
# ---------------------------------------------------------------------------
MASAKHANEWS_REPO = "masakhane/masakhanews"
MASAKHANEWS_CONFIG = {"xh": "xho", "en": "eng"}

# ---------------------------------------------------------------------------
# Orthography: word pairs reported in the EDA (historical form, modern form)
# ---------------------------------------------------------------------------
ORTHOGRAPHY_PAIRS = [
    ("apa", "apha"),
    ("wati", "wathi"),
    ("ukuti", "ukuthi"),
    ("tixo", "thixo"),
    ("pezu", "phezu"),
    ("betu", "bethu"),
    ("yetu", "yethu"),
    ("ukuteta", "ukuthetha"),
]

# Consonants after which the normaliser may restore an aspirate "h".
# Modern isiXhosa writes aspiration as ph, th, kh, ch, qh, xh.
# CONFIRM this set with an isiXhosa speaker / reference grammar before reporting.
ASPIRABLE = frozenset("ptkcqx")

# Normaliser hyper-parameter grid (tuned on the validation half of the gold set)
NORMALISER_GRID = {
    "min_freq": [1, 2, 3, 5, 10],      # candidate must occur at least this often in the modern lexicon
    "ratio": [1.0, 2.0, 5.0],          # ...and this many times more often than the original form
    "max_insert": [1, 2, 3],           # maximum number of h insertions per word
}

# ---------------------------------------------------------------------------
# Models (E1, E2) and language ID (E3)
# ---------------------------------------------------------------------------
HF_MODELS = {
    "xlmr": "FacebookAI/xlm-roberta-base",
    "afroxlmr": "Davlan/afro-xlmr-base",
}
TIKTOKEN_ENCODING = "o200k_base"
# Extra tokenisers for E1 only: repo -> trust_remote_code. Skipped with a warning if not accessible.
# InkubaLM (Lelapa AI) is trained on isiXhosa with its own vocabulary; the repo is gated, so accept its
# terms on huggingface.co and run `hf auth login` (or `huggingface-cli login`) first.
E1_EXTRA_TOKENISERS = {"inkubalm": ("lelapa/InkubaLM-0.4B", True)}
PPPL_MAX_TOKENS = 128
PPPL_BATCH_SIZE = 32
BOOTSTRAP_SAMPLES = 1000

GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model.bin"
LID_EXPECTED = {"xh": "xho_Latn", "en": "eng_Latn"}

# ---------------------------------------------------------------------------
# E4 (stretch): newspaper attribution in the overlap window
# ---------------------------------------------------------------------------
OVERLAP_YEARS = (1884, 1888)


def ensure_dirs() -> None:
    """Create all output folders (safe to call repeatedly)."""
    for d in (PROCESSED_DIR, CONDITIONS_DIR, RESULTS_DIR, FIGURES_DIR, ANNOTATIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
