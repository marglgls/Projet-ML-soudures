"""IQR (1.5x) outlier screen for every continuous column.
Lists bounds, counts below/above, and extreme values.
Run from the repo root:  python3 outlier_scan.py
"""
import json
import statistics

rows = [l.split() for l in open('data/welddb/welddb.data', encoding='utf-8') if l.strip()]
cols = json.load(open('context/columns.json', encoding='utf-8'))


def numvals(j):
    v = []
    for r in rows:
        t = r[j]
        if t == 'N':
            continue
        try:
            v.append(float(t))
        except ValueError:
            pass
    return v


for k, entry in cols.items():
    if not entry['type'].startswith('continuous'):
        continue
    v = sorted(numvals(int(k) - 1))
    n = len(v)
    if n < 8:
        continue
    q1 = statistics.median(v[:n // 2])
    q3 = statistics.median(v[(n + 1) // 2:])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    low = [x for x in v if x < lo]
    high = [x for x in v if x > hi]
    print(f"col {k} ({entry['label']}): n={n} Q1={q1:.4g} Q3={q3:.4g} "
          f"IQR={iqr:.4g} bounds=[{lo:.4g},{hi:.4g}] n_low={len(low)} n_high={len(high)}")
    if low:
        print(f"   low extremes: {low[:8]}")
    if high:
        print(f"   high extremes: {high[-8:]}")
