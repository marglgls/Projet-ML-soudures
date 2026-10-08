"""Assemble data/cleaned/cleaned.csv from every intermediate table.

Generic join: each *.csv in data/intermediate/ provides weld_id + one or more
`colNN_Label` columns built by a preprocessing step. Any column 1..43 covered
by no intermediate file falls back to the raw value from
data/welddb/welddb.data ('N' -> empty field). Columns listed in DROPPED (per
the data_preprocessing.ipynb decision) are excluded. A column covered twice is
an error. Rows are joined positionally (1652 rows, weld_id order checked
identical everywhere — weld IDs are not unique so no key-join).

Run from the repo root:  python3 preprocessing/build_model_table.py
"""
import csv
import json
import re
from pathlib import Path

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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(out_headers)
        for i, wid in enumerate(order):
            row = [wid]
            for n in kept:
                if n in covered:
                    row.append(covered[n][1][i])
                else:
                    tok = raw[i][n - 1]
                    row.append('' if tok == 'N' else tok)
            w.writerow(row)
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


if __name__ == '__main__':
    main()
