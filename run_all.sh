#!/usr/bin/env bash
# Reproduces all analyses of the revised study. Requires data/cohort.pkl (see README). Random seeds are fixed in lib.py (SEED = 42).
set -e
python final_numbers.py      # expanding-window validation, comparisons, DCA, coefficients, bootstrap, sensitivity -> final_numbers.pkl, final_preds.pkl
python extra.py              # missingness tests, complete-case analysis -> extra.pkl
python approach_recal.py     # approach-specific calibration/recalibration, Holm adjustment -> approach_recal.pkl
python make_figs.py          # Figures 1-4 -> figures/
# Online Resource 1 (initial single-split analysis):
python original_single_split/main.py; python original_single_split/main.py nores
python original_single_split/nested.py; python original_single_split/strat.py
python original_single_split/seeds.py 1; python original_single_split/seeds.py 7
