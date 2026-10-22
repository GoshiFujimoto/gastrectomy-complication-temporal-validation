"""Convert the institutional Excel database to the pickle used by the analysis (run locally; data are never committed).
usage: python prepare_data.py path/to/database.xlsx"""
import sys, pandas as pd
pd.read_excel(sys.argv[1], sheet_name=0).to_pickle('data/cohort.pkl')
