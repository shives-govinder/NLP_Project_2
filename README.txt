NLP PROJECT 2 - MODERN NLP TOOLS VS HISTORICAL ISIXHOSA
COMS4054A, 2026

1. SETUP
   python3.11 -m venv .venv
   source .venv/bin/activate          (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt

2. DATA
   Place the course data so this path exists, next to this code folder:
     ../data/TxtLangFiles1870-1894_TB/XhTxt/
     ../data/TxtLangFiles1870-1894_TB/EnTxt/
     ../data/TxtLangFiles1870-1894_TB/LanguageSplitTracker70-94.csv
   Or set the environment variable NLP2_DATA_DIR to the folder containing TxtLangFiles1870-1894_TB.
   An internet connection is needed on first run (Hugging Face downloads).

3. RUN (in this order, from this folder)
   python -m nlp2.data_prep
   python -m nlp2.modern_data
   python -m nlp2.eda
   python -m nlp2.normalise tune          (uses annotations/gold_normalisation.csv)
   python -m nlp2.normalise apply
   python -m nlp2.e1_tokeniser_cost
   python -m nlp2.e2_pseudo_perplexity    (GPU recommended)
   python -m nlp2.e3_language_id
   python -m nlp2.e4_attribution
   python -m nlp2.plots
   Or run everything with: ./run_all.sh

4. OUTPUTS
   outputs/results/   CSV results for every experiment, table1.tex
   outputs/figures/   figures used in the extended abstract

See README.md for details, design notes and expected runtimes.
