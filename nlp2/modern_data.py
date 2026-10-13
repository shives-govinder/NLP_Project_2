"""Step 2: modern reference corpora (MasakhaNEWS isiXhosa + English).

Usage:
    python -m nlp2.modern_data

Design choices (state these in the Method section):
  * Segments with web boilerplate (BBC, social-media links, 'read more') are dropped.
  * MasakhaNEWS is news text, roughly matching the genre of the historical newspapers.
  * The modern isiXhosa LEXICON for the normaliser is built from train + validation.
  * The modern CONDITIONS used in E1-E3 are sampled from the TEST split only, so the
    text we evaluate on never contributed to the lexicon.

Outputs:
    outputs/processed/modern_xh_lexicon.csv     word, freq
    outputs/processed/conditions/modern_xh.csv
    outputs/processed/conditions/modern_en.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter

import pandas as pd

from . import config
from .data_prep import WORD_RE, clean_text, finalise_segment, keep_segment, segment
from .utils import get_logger, save_condition, set_seed

log = get_logger("modern_data")

# Web boilerplate found in MasakhaNEWS (mostly the English BBC articles).
BOILERPLATE_RE = re.compile(
    r"\b(BBC|Facebook|Twitter|Instagram|WhatsApp|YouTube|subscribe|newsletter|click here|follow us|read more)\b", re.I
)
SPLIT_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}


def load_masakhanews(lang_code: str) -> dict[str, pd.DataFrame]:
    """Return {split: DataFrame} for a MasakhaNEWS language config (e.g. 'xho')."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.MASAKHANEWS_REPO, lang_code)
        return {split: ds[split].to_pandas() for split in ds}
    except Exception as exc:  # fall back to reading the TSVs directly
        log.warning("datasets.load_dataset failed (%s); falling back to raw TSV download", exc)
        from huggingface_hub import hf_hub_download

        out = {}
        for split, fname in SPLIT_FILES.items():
            path = hf_hub_download(
                config.MASAKHANEWS_REPO, f"data/{lang_code}/{fname}", repo_type="dataset"
            )
            out[split] = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_MINIMAL, on_bad_lines="warn")
        return out


def article_text(df: pd.DataFrame) -> pd.Series:
    if "headline_text" in df.columns:
        return df["headline_text"].fillna("").astype(str)
    return (df.get("headline", "").fillna("") + ". " + df["text"].fillna("")).astype(str)


def segments_from_articles(df: pd.DataFrame, lang: str, split: str) -> pd.DataFrame:
    rows = []
    for a_idx, text in enumerate(article_text(df)):
        for s_idx, seg in enumerate(segment(clean_text(text))):
            seg, _ = finalise_segment(seg)
            if keep_segment(seg) and not BOILERPLATE_RE.search(seg):
                rows.append(
                    {
                        "seg_id": f"MN_{lang}_{split}_{a_idx:04d}_{s_idx:03d}",
                        "lang": lang,
                        "split": split,
                        "text": seg,
                        "n_words": len(WORD_RE.findall(seg)),
                        "had_illegible": False,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(subset="text", keep="first").reset_index(drop=True)


def build_lexicon(texts) -> Counter:
    counts: Counter = Counter()
    for t in texts:
        counts.update(w.lower() for w in WORD_RE.findall(t))
    return counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=config.SAMPLE_PER_CONDITION)
    args = ap.parse_args(argv)
    set_seed()
    config.ensure_dirs()

    for lang, code in config.MASAKHANEWS_CONFIG.items():
        splits = load_masakhanews(code)
        log.info("%s: %s", code, {k: len(v) for k, v in splits.items()})

        if lang == "xh":
            lex = build_lexicon(pd.concat([article_text(splits["train"]), article_text(splits["validation"])]))
            pd.DataFrame(lex.most_common(), columns=["word", "freq"]).to_csv(config.LEXICON_CSV, index=False)
            log.info("Lexicon: %d types, %d tokens -> %s", len(lex), sum(lex.values()), config.LEXICON_CSV)

        test_segs = segments_from_articles(splits["test"], lang, "test")
        sample = test_segs.sample(n=min(args.sample, len(test_segs)), random_state=config.SEED)
        name = f"modern_{lang}"
        log.info("Saved %s (%d of %d test segments)", save_condition(name, sample.sort_values("seg_id")), len(sample), len(test_segs))


if __name__ == "__main__":
    main()
