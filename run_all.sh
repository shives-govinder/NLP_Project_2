#!/usr/bin/env bash
# Runs the full pipeline in order. See README.md for what each step does.
# E2 is slow on CPU; run it on a GPU (e.g. Colab) and copy outputs/results/e2/ back.
set -euo pipefail
cd "$(dirname "$0")"

python -m nlp2.data_prep
python -m nlp2.modern_data
python -m nlp2.eda

if [ -s annotations/gold_normalisation.csv ] && [ "$(wc -l < annotations/gold_normalisation.csv)" -gt 1 ]; then
  python -m nlp2.normalise tune
else
  echo "!! annotations/gold_normalisation.csv has no rows yet - normaliser will use default parameters."
fi
python -m nlp2.normalise apply

python -m nlp2.e1_tokeniser_cost
python -m nlp2.e2_pseudo_perplexity
python -m nlp2.e3_language_id
python -m nlp2.e4_attribution
python -m nlp2.plots
