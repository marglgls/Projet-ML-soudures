"""Preprocessing step 4b — unknown post-weld heat treatment (cols 29 and 30).

13 rows miss both the heat-treatment temperature and time. They are all
'Gar&K-1975-25mm-<n>ht' welds: the 'ht' suffix means heat-treated, so the
generic median imputation (250 degC, which is in fact the 250 degC / 14 h
degassing plateau) would describe them wrongly. The same article treats its
other welds ('Gar&K-1975-12mm-<n>ht') at 600 degC for 0.75 h: we assume the
25 mm welds received the same treatment. This is an assumption, documented in
the report; rows where cols 29-30 are reported are never changed.

Writes a modeling-ready table with one row per weld (1652 rows):

    data/intermediate/heat_treatment.csv  ->  weld_id + col29 + col30

Run from the repo root:  python3 preprocessing/heat_treatment.py
"""
import csv
import re
from pathlib import Path

REPO = Path.cwd()
RAW = REPO / 'data' / 'welddb' / 'welddb.data'
OUT = REPO / 'data' / 'intermediate' / 'heat_treatment.csv'
ID_COL = 44  # Weld ID, last column
TEMP_COL, TIME_COL = 29, 30
HEADERS = {TEMP_COL: 'col29_Post_weld_heat_treatment_temperature',
           TIME_COL: 'col30_Post_weld_heat_treatment_time'}
# Weld ID pattern -> (temperature degC, time h) assumed when both are missing.
ASSUMED = [(re.compile(r'^Gar&K-1975-25mm-\d+ht$'), (600.0, 0.75))]


def assumed_treatment(weld_id):
    for pattern, treatment in ASSUMED:
        if pattern.match(weld_id):
            return treatment
    return None


def number(token):
    return None if token == 'N' else float(token)


def build_table():
    rows = [l.split() for l in RAW.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert all(len(r) == 44 for r in rows), 'expected 44 whitespace-separated columns'
    table, filled = [], []
    for r in rows:
        wid = r[ID_COL - 1]
        temp, time = number(r[TEMP_COL - 1]), number(r[TIME_COL - 1])
        if temp is None and time is None:
            treatment = assumed_treatment(wid)
            if treatment is not None:
                temp, time = treatment
                filled.append(wid)
        table.append({
            'weld_id': wid,
            HEADERS[TEMP_COL]: '' if temp is None else repr(temp),
            HEADERS[TIME_COL]: '' if time is None else repr(time),
        })
    return table, filled


def main():
    table, filled = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    still_missing = sum(1 for t in table if t[HEADERS[TEMP_COL]] == '')
    print(f'Unknown treatment filled from the same article (600 degC, 0.75 h): '
          f'{len(filled)} rows ({filled[0]} ... {filled[-1]}); still missing: {still_missing}')
    print(f'Wrote {OUT.relative_to(REPO)}: {len(table)} rows x {len(table[0])} columns')
    return filled


if __name__ == '__main__':
    main()
