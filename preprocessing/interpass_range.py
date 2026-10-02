"""Preprocessing step 3 — interpass 'xx-yy' range token (col 27).

38 rows report the interpass temperature as the range '150-200' instead of a
single value (see context/columns.json col 27 note). This step replaces every
'xx-yy' token with the mean of the range ('150-200' -> 175.0) and writes a
modeling-ready table with one row per weld (1652 rows):

    data/intermediate/interpass_mean.csv  ->  weld_id + col27_Interpass_temperature

'N' (not reported) would become an empty field (col 27 has none). Plain numeric
values pass through unchanged. The raw file is never modified.

Run from the repo root:  python3 preprocessing/interpass_range.py
"""
import csv
import json
import re
from pathlib import Path

REPO = Path.cwd()
OUT = REPO / 'data' / 'intermediate' / 'interpass_mean.csv'
ID_COL = 44  # Weld ID, last column
RANGE_RE = re.compile(r'^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$')


def parse(token):
    """'N' -> None (missing); 'xx-yy' -> mean(xx, yy); otherwise float(token)."""
    if token == 'N':
        return None
    m = RANGE_RE.match(token)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    return float(token)


def build_table(repo=REPO):
    schema = json.loads((repo / 'context' / 'columns.json').read_text(encoding='utf-8'))
    numeric_cols = [int(n) for n, m in schema.items()
                    if m['type'].startswith('continuous')]
    rows = [(l.split()) for l in (repo / 'data' / 'welddb' / 'welddb.data')
            .read_text(encoding='utf-8').splitlines() if l.strip()]
    assert all(len(r) == 44 for r in rows), 'expected 44 whitespace-separated columns'
    # Concerned columns: numerical columns actually containing a range token.
    concerned = [n for n in numeric_cols
                 if any(RANGE_RE.match(r[n - 1]) for r in rows)]
    table, counts, examples = [], {}, []
    for r in rows:
        record = {'weld_id': r[ID_COL - 1]}
        for num in concerned:
            raw = r[num - 1]
            try:
                val = parse(raw)
            except ValueError:
                raise ValueError(f'unexpected token {raw!r} in column {num}')
            if RANGE_RE.match(raw):
                counts[raw] = counts.get(raw, 0) + 1
                if len(examples) < 8:
                    examples.append((r[ID_COL - 1], raw, val))
            record[f"col{num:02d}_Interpass_temperature"] = '' if val is None else repr(val)
        table.append(record)
    return concerned, table, counts, examples


def main():
    concerned, table, counts, examples = build_table()
    headers = [k for k in table[0] if k != 'weld_id']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['weld_id'] + headers)
        w.writeheader()
        w.writerows(table)
    total = sum(counts.values())
    print(f'Concerned columns: {concerned} (tokens matching xx-yy)')
    print('Replacements xx-yy -> mean: ' +
          ', '.join(f'{t} -> {parse(t)!r} (x{c})' for t, c in sorted(counts.items())) +
          f' (total {total})')
    print(f'Wrote {OUT.relative_to(REPO)}: {len(table)} rows x {len(headers) + 1} columns')
    print('Examples (weld_id | raw -> new):')
    for wid, raw, val in examples:
        print(f'  {wid} | {raw} -> {val!r}')
    return counts


if __name__ == '__main__':
    main()
