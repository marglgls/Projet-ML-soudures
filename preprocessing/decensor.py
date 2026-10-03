"""Preprocessing step 1 — left-censored '<...' values.

In numerical columns, a token like '<0.002' means 'below the detection limit
0.002' (see context/columns.json 'censored_tokens'). This step replaces every
'<x' token with the numeric value x, and writes a modeling-ready table with
one row per weld (1652 rows):

    data/intermediate/decensored.csv  ->  weld_id + one column per concerned column

'N' (not reported) becomes an empty field. Plain numeric values pass through
unchanged. The raw file is never modified.

Run from the repo root:  python3 preprocessing/decensor.py
"""
import csv
import json
from pathlib import Path

REPO = Path.cwd()
RAW = REPO / 'data' / 'welddb' / 'welddb.data'
SCHEMA = REPO / 'context' / 'columns.json'
OUT = REPO / 'data' / 'intermediate' / 'decensored.csv'
ID_COL = 44  # Weld ID, last column
# Unit fixes for censored tokens reported in another unit than the column.
# Col 9 (V, wt%): the 9 'EvansLetter...Mo' rows give '<5', like their ppm
# columns (Ti, Al, B, Nb all '<5' on the same rows): it is 5 ppmw, not 5 wt%
# (max V otherwise 0.32 wt%). 5 ppmw = 0.0005 wt%, the column's usual limit.
# Cols 14 and 16 (Ti, Al, ppmw): the 16 'PantK-1990' rows give '<0.01', a wt%
# limit (a 0.01 ppmw detection limit is unrealistic; Nb on the same rows is
# 200-900 ppmw). 0.01 wt% = 100 ppmw, a limit already common in these columns.
UNIT_FIX = {(9, '<5'): 0.0005, (14, '<0.01'): 100.0, (16, '<0.01'): 100.0}


def concerned_columns(schema):
    """Numerical columns whose schema documents censored '<...' tokens."""
    out = []
    for num, meta in sorted(schema.items(), key=lambda kv: int(kv[0])):
        if meta['type'].startswith('continuous') and 'censored_tokens' in meta['observed']:
            label = meta['label'].replace(' ', '_')
            out.append((int(num), f"col{int(num):02d}_{label}"))
    return out


def decensor(token):
    """'N' -> None (missing); '<x' -> float(x); otherwise float(token)."""
    if token == 'N':
        return None
    if token.startswith('<'):
        return float(token[1:])
    return float(token)


def build_table(repo=REPO):
    schema = json.loads((repo / 'context' / 'columns.json').read_text(encoding='utf-8'))
    cols = concerned_columns(schema)
    rows = [(l.split()) for l in (repo / 'data' / 'welddb' / 'welddb.data')
            .read_text(encoding='utf-8').splitlines() if l.strip()]
    assert all(len(r) == 44 for r in rows), 'expected 44 whitespace-separated columns'
    table, counts, examples = [], {}, []
    for r in rows:
        record = {'weld_id': r[ID_COL - 1]}
        for num, header in cols:
            raw = r[num - 1]
            try:
                val = UNIT_FIX.get((num, raw), None) or decensor(raw)
            except ValueError:
                raise ValueError(f'unexpected token {raw!r} in column {num}')
            if raw.startswith('<'):
                counts[num] = counts.get(num, 0) + 1
                if len(examples) < 8:
                    examples.append((r[ID_COL - 1], header, raw, val))
            record[header] = '' if val is None else repr(val)
        table.append(record)
    return [c[1] for c in cols], table, counts, examples


def main():
    headers, table, counts, examples = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['weld_id'] + headers)
        w.writeheader()
        w.writerows(table)
    total = sum(counts.values())
    print(f'Concerned columns ({len(headers)}): ' + ', '.join(str(n) for n in sorted(counts)))
    print(f"Replacements '<x' -> x per column: " +
          ', '.join(f"col{n:02d}={counts[n]}" for n in sorted(counts)) +
          f' (total {total})')
    print(f'Wrote {OUT.relative_to(REPO)}: {len(table)} rows x {len(headers) + 1} columns')
    for (num, raw), val in UNIT_FIX.items():
        print(f'Unit fix: col{num:02d} {raw!r} -> {val!r} (limit given in another unit, see UNIT_FIX)')
    print('Examples (weld_id | column | raw -> new):')
    for wid, header, raw, val in examples:
        print(f'  {wid} | {header} | {raw} -> {val!r}')
    return counts


if __name__ == '__main__':
    main()
