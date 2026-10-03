"""Preprocessing step 4 — categorical columns: current type (col 24) and weld
process (col 28).

col 24 'AC or DC' is missing on 215 rows, but the electrode polarity (col 25)
tells the current type: '+' / '-' only exist in DC, '0' is the AC code (see
context/columns.json col 25 note). This step fills col 24 from col 25 when
col 24 is missing ('+'/'-' -> DC, '0' -> AC); 156 rows miss both and stay
missing. Rows where col 24 is reported are never changed.

col 28 'Type of weld' has synonyms and very rare codes (see col 28 note):
  - ShMA is the same process as MMA (manual metal arc)        -> MMA
  - SAA (4 rows, undocumented SA variant), NGSAW (submerged
    arc narrow gap, 18 rows)                                  -> SA
  - GTAA, GMAA, NGGMA (gas-shielded arc, 4 + 4 + 7 rows)      -> GSA
TSA (tandem submerged arc, 87 rows) and FCA (87 rows) are frequent enough to
stay as their own levels. Result: 5 levels MMA, SA, TSA, FCA, GSA.

Writes a modeling-ready table with one row per weld (1652 rows):

    data/intermediate/categorical.csv  ->  weld_id + col24_AC_or_DC + col28_Type_of_weld

'N' (not reported) becomes an empty field. The raw file is never modified.

Run from the repo root:  python3 preprocessing/categorical_clean.py
"""
import csv
from collections import Counter
from pathlib import Path

REPO = Path.cwd()
RAW = REPO / 'data' / 'welddb' / 'welddb.data'
OUT = REPO / 'data' / 'intermediate' / 'categorical.csv'
ID_COL = 44  # Weld ID, last column
CURRENT_COL, POLARITY_COL, PROCESS_COL = 24, 25, 28
POLARITY_TO_CURRENT = {'+': 'DC', '-': 'DC', '0': 'AC'}
PROCESS_GROUP = {
    'MMA': 'MMA', 'ShMA': 'MMA',
    'SA': 'SA', 'SAA': 'SA', 'NGSAW': 'SA',
    'TSA': 'TSA',
    'FCA': 'FCA',
    'GTAA': 'GSA', 'GMAA': 'GSA', 'NGGMA': 'GSA',
}


def current_type(current, polarity):
    """Reported col 24 kept; missing col 24 inferred from polarity if known."""
    if current != 'N':
        return current
    return POLARITY_TO_CURRENT.get(polarity)


def process_group(process):
    try:
        return PROCESS_GROUP[process]
    except KeyError:
        raise ValueError(f'unexpected weld process {process!r} in column {PROCESS_COL}')


def build_table():
    rows = [l.split() for l in RAW.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert all(len(r) == 44 for r in rows), 'expected 44 whitespace-separated columns'
    table, inferred, merged = [], Counter(), Counter()
    for r in rows:
        current, polarity = r[CURRENT_COL - 1], r[POLARITY_COL - 1]
        process = r[PROCESS_COL - 1]
        new_current = current_type(current, polarity)
        if current == 'N' and new_current is not None:
            inferred[f'{polarity} -> {new_current}'] += 1
        new_process = process_group(process)
        if new_process != process:
            merged[f'{process} -> {new_process}'] += 1
        table.append({
            'weld_id': r[ID_COL - 1],
            'col24_AC_or_DC': new_current or '',
            'col28_Type_of_weld': new_process,
        })
    return table, inferred, merged


def main():
    table, inferred, merged = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    still_missing = sum(1 for t in table if t['col24_AC_or_DC'] == '')
    print('col 24 filled from polarity: ' +
          ', '.join(f'{k} (x{v})' for k, v in sorted(inferred.items())) +
          f' (total {sum(inferred.values())}); still missing: {still_missing}')
    print('col 28 merged: ' +
          ', '.join(f'{k} (x{v})' for k, v in sorted(merged.items())) +
          f' (total {sum(merged.values())})')
    print('col 28 levels: ' + ', '.join(
        f'{k}={v}' for k, v in Counter(t['col28_Type_of_weld'] for t in table).most_common()))
    print(f'Wrote {OUT.relative_to(REPO)}: {len(table)} rows x {len(table[0])} columns')
    return inferred, merged


if __name__ == '__main__':
    main()
