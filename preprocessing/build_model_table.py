"""Assemble data/cleaned/cleaned.csv from every intermediate table.

Generic join: each *.csv in data/intermediate/ provides weld_id + one or more
`colNN_Label` columns built by a preprocessing step. Any column 1..43 covered
by no intermediate file falls back to the raw value from
data/welddb/welddb.data ('N' -> empty field). Columns listed in DROPPED (per
the data_preprocessing.ipynb decision) are excluded. A column covered twice is
an error. Rows are joined positionally (1652 rows, weld_id order checked
identical everywhere — weld IDs are not unique so no key-join).

Step 5b (alloy_zero.py) is applied to the assembled table before writing:
Ni, Cr, Mo, V, Nb never reported by an article -> 0.

Charpy tests in BAD_CHARPY are emptied (temperature and energy) before writing:
the row then reports no test result and is used by no model.

Run from the repo root:  python3 preprocessing/build_model_table.py
"""
import csv
import json
import re
from pathlib import Path

from alloy_zero import fill_zero, report

REPO = Path.cwd()
INTER = REPO / 'data' / 'intermediate'
RAW = REPO / 'data' / 'welddb' / 'welddb.data'
SCHEMA = REPO / 'context' / 'columns.json'
OUT = REPO / 'data' / 'cleaned' / 'cleaned.csv'
HEADER_RE = re.compile(r'^col(\d+)_(.+)$')
# Columns ignored per the data_preprocessing.ipynb decision: targets too sparse
# to predict (37-43: hardness, FATT, microstructure) + input features half-empty
# or more (10, 11, 12, 17, 19, 20, 21). Intermediates still compute them; they
# are excluded here at assembly time.
DROPPED = {10, 11, 12, 17, 19, 20, 21, 37, 38, 39, 40, 41, 42, 43}
# Output column order: the inputs (1-30), then the Charpy test temperature (35),
# a test condition used as an input of the Charpy model, then the outputs.
ORDER = list(range(1, 31)) + [35] + [n for n in range(31, 44) if n != 35]
# Charpy tests removed as data-entry errors (temperature col 35 and energy col 36
# emptied). Ditt-0.5rch2: 188 degC, the only Charpy temperature above 70 degC;
# the 4 other welds of the article are all tested at 20, -10 and -30 degC, and
# this weld at 20, 188 and -30 degC: almost certainly a typo for -10 degC.
BAD_CHARPY = {'Ditt-0.5rch2'}
CHARPY_COLS = ('col35_Charpy_temperature', 'col36_Charpy_impact_toughness')


def main():
    schema = json.loads(SCHEMA.read_text(encoding='utf-8'))
    schema_headers = {int(n): f"col{int(n):02d}_{m['label'].replace(' ', '_')}"
                      for n, m in schema.items() if n != '44'}
    raw = [l.split() for l in RAW.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(raw) == 1652 and all(len(r) == 44 for r in raw)
    raw_ids = [r[43] for r in raw]

    inputs = sorted(p for p in INTER.glob('*.csv') if p.name != OUT.name)
    assert inputs, f'no intermediate files in {INTER}'
    covered, order = {}, None
    for path in inputs:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            assert 'weld_id' in reader.fieldnames, path.name
            recs = list(reader)
        ids = [r['weld_id'] for r in recs]
        order = ids if order is None else order
        assert ids == order, f'row order mismatch in {path.name}'
        for header in reader.fieldnames[1:]:
            m = HEADER_RE.match(header)
            assert m, f'bad header {header!r} in {path.name}'
            num = int(m.group(1))
            assert num not in covered, f'column {num} covered by two files'
            assert header == schema_headers[num], f'header mismatch {header!r} in {path.name}'
            covered[num] = (path.name, [r[header] for r in recs])
    assert raw_ids == order, 'intermediate row order differs from raw file'

    kept = [n for n in ORDER if n not in DROPPED]
    out_headers = ['weld_id'] + [schema_headers[n] for n in kept]
    provenance = {}
    table = []
    for i, wid in enumerate(order):
        row = [wid]
        for n in kept:
            if n in covered:
                row.append(covered[n][1][i])
            else:
                tok = raw[i][n - 1]
                row.append('' if tok == 'N' else tok)
        table.append(row)

    # Step 5b: alloying elements never reported by an article -> 0 (alloy_zero.py)
    filled = fill_zero(out_headers, table)

    # Charpy tests removed as data-entry errors (BAD_CHARPY)
    removed = [row for row in table if row[0] in BAD_CHARPY]
    assert len(removed) == len(BAD_CHARPY), 'BAD_CHARPY weld not found'
    for row in removed:
        for h in CHARPY_COLS:
            row[out_headers.index(h)] = ''

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(out_headers)
        w.writerows(table)
    for n in range(1, 44):
        if n not in DROPPED:
            provenance[n] = covered[n][0] if n in covered else 'raw'
    from collections import Counter
    print(f'Intermediate files joined ({len(inputs)}): ' + ', '.join(p.name for p in inputs))
    for src, c in sorted(Counter(provenance.values()).items()):
        cols = sorted(n for n, s in provenance.items() if s == src)
        print(f'  {src}: {c} columns {cols}')
    print(f'Wrote {OUT.relative_to(REPO)}: 1652 rows x {len(out_headers)} columns')
    print(f'Dropped per notebook decision ({len(DROPPED)}): {sorted(DROPPED)}')
    report(filled)
    print(f'Charpy tests removed as data-entry errors: {sorted(BAD_CHARPY)}')


if __name__ == '__main__':
    main()
