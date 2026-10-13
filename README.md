# NLP Project 2 — Modern NLP tools vs historical isiXhosa

COMS4054A Natural Language Processing, 2026 · Wits University
Data: [isiXhosa Intellectual Traditions archive](https://ibali.uct.ac.za/s/isixit/page/welcome) (Schoots et al., 2023)

**Research question.** Do modern multilingual language technologies systematically underserve
19th-century isiXhosa text, and how much of that gap is explained by orthographic convention
rather than the language itself?

- **H1** XLM-R / AfroXLMR show higher tokeniser fertility and pseudo-perplexity on historical than
  modern isiXhosa, and the gap is larger than the historical-vs-modern English gap.
- **H2** A lexicon-gated aspirate-*h* normaliser closes a measurable share of that gap.
- **H3** Off-the-shelf language ID recognises historical isiXhosa less reliably than modern isiXhosa.

## Repository layout

```
NLP_Assignment_2_Code/
├── nlp2/                     # all code, run as modules: python -m nlp2.<name>
│   ├── config.py             # paths, seeds, sample sizes, model ids, grids
│   ├── utils.py              # seeding, condition I/O, bootstrap CIs
│   ├── data_prep.py          # step 1: clean, segment, edition-level splits, samples
│   ├── modern_data.py        # step 2: MasakhaNEWS xho/eng + modern lexicon
│   ├── normalise.py          # step 3: gold-set candidates, tuning, apply
│   ├── eda.py                # corpus statistics + figure
│   ├── e1_tokeniser_cost.py  # E1 fertility, % words split 3+, unk rate
│   ├── e2_pseudo_perplexity.py  # E2 masked-LM pseudo-perplexity (GPU)
│   ├── e3_language_id.py     # E3 GlotLID recall + confusions
│   ├── e4_attribution.py     # E4 (stretch) Isigidimi vs Imvo, 1884-1888
│   └── plots.py              # Figure 1 + Table 1 for the paper
├── annotations/              # gold set for the normaliser (committed)
├── tests/                    # pytest unit tests
├── outputs/                  # generated: processed/, results/, figures/ (git-ignored)
├── notebooks/                # scratch exploration
├── run_all.sh                # whole pipeline in order
├── requirements.txt / requirements-dev.txt
└── pyproject.toml            # pytest + ruff config
```

## Setup

Python 3.11 is recommended (3.10–3.12 work; some wheels lag behind the newest Python).

```bash
cd NLP_Assignment_2_Code
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                      # 14 unit tests, < 1 s
```

VS Code picks up `.venv` automatically (`.vscode/settings.json`); the Run and Debug panel has one
launch configuration per pipeline step.

### Data

The raw data is **not** in this repo. By default the code expects it next to this folder:

```
NLP Project 2/
├── data/TxtLangFiles1870-1894_TB/{XhTxt/, EnTxt/, LanguageSplitTracker70-94.csv}
└── NLP_Assignment_2_Code/
```

Elsewhere? `export NLP2_DATA_DIR=/path/to/data`. Outputs can be redirected with `NLP2_OUT_DIR`.
Steps 2, E1, E2 and E3 download from Hugging Face on first run (MasakhaNEWS, XLM-R, AfroXLMR, GlotLID).

## Running the pipeline

| Step | Command | Where | Writes |
| --- | --- | --- | --- |
| 1 | `python -m nlp2.data_prep` | laptop, ~20 s | `processed/segments_*.csv.gz`, `edition_splits.csv`, `conditions/hist_*.csv` |
| 2 | `python -m nlp2.modern_data` | laptop | `processed/modern_xh_lexicon.csv`, `conditions/modern_*.csv` |
| 3a | `python -m nlp2.normalise candidates` | laptop | `annotations/gold_candidates_to_annotate.csv` |
| — | annotate → `annotations/gold_normalisation.csv` | people | see `annotations/README.md` |
| 3b | `python -m nlp2.normalise tune` | laptop | `results/normaliser_grid.csv`, `normaliser_test.csv`, best params |
| 3c | `python -m nlp2.normalise apply` | laptop | `conditions/hist_xh_norm.csv` |
| EDA | `python -m nlp2.eda` | laptop | `results/eda_*.csv`, `figures/eda_words_per_year.png` |
| E1 | `python -m nlp2.e1_tokeniser_cost` | laptop | `results/e1_tokeniser_cost.csv` |
| E2 | `python -m nlp2.e2_pseudo_perplexity` | **GPU** (Colab) | `results/e2/*.csv` (cache), `results/e2_pseudo_perplexity.csv` |
| E3 | `python -m nlp2.e3_language_id` | laptop | `results/e3_language_id.csv`, `e3_confusions.csv` |
| E4 | `python -m nlp2.e4_attribution` | laptop | `results/e4_*.csv` |
| Paper | `python -m nlp2.plots` | laptop | `figures/fig1_tokeniser_and_pppl.png`, `results/table1.tex` |

Or run everything: `./run_all.sh`.

**E2 on Colab.** Upload this folder (or clone the repo) plus `outputs/processed/conditions/`, then
`pip install -r requirements.txt` and `python -m nlp2.e2_pseudo_perplexity`. Try `--limit 20` first.
Scores are checkpointed every 100 segments in `outputs/results/e2/`, so a disconnected run resumes.

## Methodology notes (for the write-up)

- **Splits are by edition**, never by sentence, so text from one issue cannot leak across splits.
- **Raw vs normalised** historical isiXhosa are the *same segments*; spelling is the only variable.
- **Same segment filter for every condition**: at least 5 words, at most 30% non-alphabetic characters, and no
  sentence end followed by a lowercase fragment (cut-off text or a lost drop-cap letter, e.g. "warns. rain drivers").
  MasakhaNEWS segments containing web boilerplate (BBC, social-media links) are also dropped.
- **Exact duplicate segments are removed** (repeated adverts, mastheads, colophons: ~18% of isiXhosa
  segments, mostly Imvo pages 1 and 4). Sentences are not split after titles or initials ("u Mr. Jabavu").
- **Length is controlled** in E3: recall is also reported per length bin and re-weighted to the modern
  sample's length distribution; raw vs normalised is compared with a paired McNemar test.
- **English is the control language**: historical Imvo English vs modern MasakhaNEWS English.
- **Modern conditions come from the MasakhaNEWS test split**; the normaliser's lexicon uses
  train + validation only.
- **Clean-subset control**: every E1–E3 result is also reported on segments with no illegible
  (`??`) words, to separate OCR noise from spelling.
- **Pseudo-perplexity** is compared across conditions *within* a model only; `pppl_word` and
  `ratio_to_modern` are the cross-model-safe views.
- All randomness is seeded (`config.SEED = 42`); CIs are 95% percentile bootstraps over segments.

## Status (13 Oct 2026)

| Module | State |
| --- | --- |
| data_prep, eda, normalise, e4 (TF-IDF), plots | Run on the full corpus; unit-tested |
| modern_data, e1, e2, e3 | Written, not yet run (need Hugging Face access) |
| Gold set | 300 candidate words generated; annotation needed |
| e4 transformer fine-tuning | TODO, stretch goal |

## Git

```bash
git init && git add . && git commit -m "Initial project scaffold"
git remote add origin <your-repo-url> && git push -u origin main
```

`outputs/` and the raw data are git-ignored. Force-add final figures if you want them in the repo:
`git add -f outputs/figures/fig1_tokeniser_and_pppl.png`.

## References

- Schoots, Ngwendu, De Wet & Muftic (2023). *IsiXhosa Intellectual Traditions Digital Archive.* RAIL 2023.
- Salazar et al. (2020). *Masked Language Model Scoring.* ACL 2020.
- Alabi et al. (2022). *Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning.* COLING 2022.
- Adelani et al. (2023). *MasakhaNEWS.* · Kargaran et al. (2023). *GlotLID.*
