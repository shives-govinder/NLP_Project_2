"""Step 1: load, clean, segment, split and sample the historical newspapers.

Usage:
    python -m nlp2.data_prep                 # full corpus + default sample sizes
    python -m nlp2.data_prep --sample 500    # smaller samples for a quick run

Outputs (in outputs/processed/):
    segments_xh.csv.gz, segments_en.csv.gz   all cleaned segments with metadata
    edition_splits.csv                       edition -> train/validation/test
    conditions/hist_xh_raw.csv               stratified sample for E1-E3
    conditions/hist_en.csv                   stratified sample for E1-E3
    cleaning_stats.csv                       what cleaning removed (for the write-up)
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, save_condition, set_seed

log = get_logger("data_prep")

FILENAME_RE = re.compile(r"^(?P<paper>IMVO|ISIG)_(?P<date>\d{4}-\d{2}-\d{2})_p(?P<page>\d{3})\.txt$")
PAPER_NAMES = {"IMVO": "Imvo Zabantsundu", "ISIG": "Isigidimi sama-Xosa"}

# Running headers such as "IMVO ZABANTSUNDU (NATIVE OPINION) 3" or
# "ISIGIDIMI SAMAXOSA, FEBRUARY 4, 1871." and lone page numbers.
HEADER_RE = re.compile(r"^\s*(\d{1,3}\s+)?(IMVO|ISIGIDIMI|NATIVE OPINION|THE KAFFIR EXPRESS)\b.*$", re.I)
PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")
ILLEGIBLE_RE = re.compile(r"\S*\?{2,}\S*")          # words containing "??" (illegible chars)
HYPHEN_BREAK_RE = re.compile(r"(\w)[-\u00ad]\s*\n\s*([a-z])")  # "word-\nrest"; keeps "u-\nMxosa"
HYPHEN_KEEP_RE = re.compile(r"(\w)-\s*\n\s*([A-Z])")  # "u-\nMxosa" -> "u-Mxosa"
HYPHEN_SPACE_RE = re.compile(r"(\w)- +([a-z])")      # "ba- behle" left by reflowed columns
# A split after one of these is undone: titles, initials and common abbreviations ("u Mr. Jabavu").
ABBREV_END_RE = re.compile(
    r"(?:\b(?:Mr|Mrs|Messrs|Rev|Revs|Dr|St|Sr|Jr|No|Nos|Sir|Hon|Esq|Capt|Col|Gen|Lt|Maj|Prof|Co|Ltd|Vol|vs|viz|etc"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)|\b[A-Z])\.[\"'\u201d]?$"
)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-Z])")
QUOTE_MAP = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
# A sentence end followed by a lowercase word fragment: text was cut or a drop-cap letter was lost
# (e.g. "...warns. rain drivers are"). Applied identically to every condition.
BROKEN_RE = re.compile(r"[.!?] [a-z]")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass
class CleanStats:
    pages: int = 0
    headers_removed: int = 0
    illegible_words: int = 0
    hyphen_joins: int = 0
    duplicates_removed: int = 0


def parse_filename(name: str) -> dict:
    """IMVO_1893-03-09_p004.txt -> paper, date, year, page, edition."""
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected filename: {name}")
    d = m.groupdict()
    return {
        "file": name,
        "paper": d["paper"],
        "paper_name": PAPER_NAMES[d["paper"]],
        "date": d["date"],
        "year": int(d["date"][:4]),
        "page": int(d["page"]),
        "edition": f"{d['paper']}_{d['date']}",
    }


def clean_text(text: str, stats: CleanStats | None = None) -> str:
    """Clean one page. Returns paragraphs separated by blank lines.

    Illegible words are replaced by config.ILLEGIBLE_TOKEN so that segments can be
    flagged later (for the 'clean subset' control) before the sentinel is removed.
    """
    stats = stats or CleanStats()
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    text = text.translate(QUOTE_MAP)  # curly quotes -> straight, same for every condition

    kept_lines = []
    for line in text.split("\n"):
        if PAGE_NUM_RE.match(line) or (HEADER_RE.match(line) and len(line.strip()) < 90):
            stats.headers_removed += 1
            continue
        kept_lines.append(line)
    text = "\n".join(kept_lines)

    text, n = HYPHEN_BREAK_RE.subn(r"\1\2", text)
    stats.hyphen_joins += n
    text = HYPHEN_KEEP_RE.sub(r"\1-\2", text)
    text, n = HYPHEN_SPACE_RE.subn(r"\1\2", text)
    stats.hyphen_joins += n
    text = text.replace("­", "")  # soft hyphens inside words

    text, n = ILLEGIBLE_RE.subn(f" {config.ILLEGIBLE_TOKEN} ", text)
    stats.illegible_words += n

    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


def segment(cleaned: str, max_words: int = config.MAX_WORDS) -> list[str]:
    """Paragraphs -> sentences; very long sentences are chunked to max_words."""
    out = []
    for para in cleaned.split("\n\n"):
        sents: list[str] = []
        for piece in SENT_SPLIT_RE.split(para):
            if sents and ABBREV_END_RE.search(sents[-1]):
                sents[-1] = f"{sents[-1]} {piece}"  # undo a split after "Mr." / "J."
            else:
                sents.append(piece)
        for sent in sents:
            words = sent.split()
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words]).strip()
                if chunk:
                    out.append(chunk)
    return out


def nonalpha_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 1.0
    return sum(not c.isalpha() for c in chars) / len(chars)


def finalise_segment(seg: str) -> tuple[str, bool]:
    """Strip the illegible sentinel; return (text, had_illegible)."""
    had = config.ILLEGIBLE_TOKEN in seg
    seg = re.sub(r"\s+", " ", seg.replace(config.ILLEGIBLE_TOKEN, " ")).strip()
    return seg, had


def keep_segment(seg: str) -> bool:
    return (
        len(WORD_RE.findall(seg)) >= config.MIN_WORDS
        and nonalpha_ratio(seg) <= config.MAX_NONALPHA_RATIO
        and not BROKEN_RE.search(seg)
    )


def build_segments(lang_dir: Path, lang: str) -> tuple[pd.DataFrame, CleanStats]:
    """Read every page in lang_dir and return one row per kept segment."""
    files = sorted(p for p in lang_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {lang_dir}. Check NLP2_DATA_DIR / config.DATA_DIR.")
    stats = CleanStats()
    rows = []
    for path in files:
        meta = parse_filename(path.name)
        raw = path.read_text(encoding="utf-8", errors="replace")
        stats.pages += 1
        for i, seg in enumerate(segment(clean_text(raw, stats))):
            text, had_ill = finalise_segment(seg)
            if not keep_segment(text):
                continue
            rows.append(
                {
                    "seg_id": f"{meta['edition']}_p{meta['page']:03d}_{lang}_{i:04d}",
                    **{k: meta[k] for k in ("paper", "date", "year", "page", "edition")},
                    "lang": lang,
                    "text": text,
                    "n_words": len(WORD_RE.findall(text)),
                    "had_illegible": had_ill,
                }
            )
    df = pd.DataFrame(rows)
    # Mastheads, colophons and adverts repeat every week ("Printed for the Proprietor, ...").
    # Keep the first occurrence of each exact segment so repeats do not dominate the samples.
    n_before = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    stats.duplicates_removed = n_before - len(df)
    # OCR-noise proxy: share of a segment's words that occur only once in the whole corpus
    # (5M+ words, so a hapax is often an OCR error). "clean": see config.CLEAN_MAX_NOISE.
    freq: dict[str, int] = {}
    for t in df["text"]:
        for w in WORD_RE.findall(t.lower()):
            freq[w] = freq.get(w, 0) + 1
    df["noise_score"] = [
        sum(freq[w] == 1 for w in ws) / max(len(ws), 1) for ws in (WORD_RE.findall(t.lower()) for t in df["text"])
    ]
    df["clean"] = (df["noise_score"] <= config.CLEAN_MAX_NOISE) & ~df["had_illegible"]
    log.info("%s: %d pages -> %d segments (%.0f%% clean)", lang, stats.pages, len(df), 100 * df["clean"].mean())
    return df, stats


def assign_edition_splits(editions: list[str], seed: int = config.SEED) -> pd.DataFrame:
    """Split by EDITION so one issue never sits on both sides of a split.

    Stratified by paper so both newspapers appear in every split.
    """
    rng = np.random.default_rng(seed)
    train_f, val_f, _ = config.SPLIT_FRACTIONS
    rows = []
    eds = pd.Series(sorted(set(editions)))
    for paper, group in eds.groupby(eds.str[:4]):
        shuffled = group.to_numpy().copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train, n_val = int(round(train_f * n)), int(round(val_f * n))
        for j, ed in enumerate(shuffled):
            split = "train" if j < n_train else "validation" if j < n_train + n_val else "test"
            rows.append({"edition": ed, "paper": paper, "split": split})
    return pd.DataFrame(rows)


def stratified_sample(df: pd.DataFrame, n: int, by: list[str], seed: int = config.SEED) -> pd.DataFrame:
    """Proportional stratified sample of n rows (or all rows if fewer)."""
    if len(df) <= n:
        return df.copy()
    alloc = (df.groupby(by).size() / len(df) * n).round().astype(int).clip(lower=1)
    parts = [
        g.sample(n=min(len(g), alloc.loc[key]), random_state=seed)
        for key, g in df.groupby(by)
    ]
    out = pd.concat(parts)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.sort_values("seg_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=config.SAMPLE_PER_CONDITION, help="segments per condition")
    args = ap.parse_args(argv)

    set_seed()
    config.ensure_dirs()

    xh, xh_stats = build_segments(config.XH_DIR, "xh")
    en, en_stats = build_segments(config.EN_DIR, "en")

    splits = assign_edition_splits(list(xh["edition"]) + list(en["edition"]))
    splits.to_csv(config.PROCESSED_DIR / "edition_splits.csv", index=False)
    split_map = dict(zip(splits["edition"], splits["split"]))
    for df in (xh, en):
        df["split"] = df["edition"].map(split_map)

    xh.to_csv(config.PROCESSED_DIR / "segments_xh.csv.gz", index=False)
    en.to_csv(config.PROCESSED_DIR / "segments_en.csv.gz", index=False)

    pd.DataFrame(
        [{"lang": "xh", **vars(xh_stats), "segments": len(xh)}, {"lang": "en", **vars(en_stats), "segments": len(en)}]
    ).to_csv(config.PROCESSED_DIR / "cleaning_stats.csv", index=False)

    # Samples for E1-E3. English is almost entirely from Imvo (Isigidimi has ~20k
    # English words), so the English condition is effectively "historical Imvo English".
    xh_sample = stratified_sample(xh, args.sample, by=["paper", "year"])
    en_sample = stratified_sample(en[en["paper"] == "IMVO"], args.sample, by=["year"])
    log.info("Saved %s", save_condition("hist_xh_raw", xh_sample))
    log.info("Saved %s", save_condition("hist_en", en_sample))


if __name__ == "__main__":
    main()
