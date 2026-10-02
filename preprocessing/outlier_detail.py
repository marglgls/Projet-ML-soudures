"""Value-count distributions at the extremes of selected columns:
shows whether a tail is continuous, stepped (series), or isolated singles.
Run from the repo root:  python3 outlier_detail.py
"""
from collections import Counter

rows = [l.split() for l in open('data/welddb/welddb.data', encoding='utf-8') if l.strip()]


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


def show(k, lo_n=12, hi_n=12):
    v = numvals(int(k) - 1)
    c = Counter(v)
    keys = sorted(c)
    print(f"--- col {k}: n={len(v)} nunique={len(keys)}")
    print("  lowest:", [(x, c[x]) for x in keys[:lo_n]])
    print("  highest:", [(x, c[x]) for x in keys[-hi_n:]])


for k in ['1', '2', '3', '4', '5', '6', '10', '11', '12', '13', '19',
          '22', '23', '26', '31', '32', '34', '35', '36', '40',
          '20', '21', '15', '16', '18']:
    show(k)
