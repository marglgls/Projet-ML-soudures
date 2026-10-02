"""Find the rows / Weld IDs holding extreme values, to check whether
outliers are shared rows, same-weld pairs, or source-level patterns.
Run from the repo root:  python3 outlier_rows.py
"""
rows = [l.split() for l in open('data/welddb/welddb.data', encoding='utf-8') if l.strip()]


def find(col, val):
    j = col - 1
    return [(i, r[43]) for i, r in enumerate(rows) if r[j] == val]


print('YS 920 rows:', find(31, '920'))
print('UTS 1151 rows:', find(32, '1151'))
print('CharpyT 188 rows:', find(35, '188'))
print('CharpyE 270 rows:', find(36, '270'))
print('O 1650 rows:', find(13, '1650'))
print('Sn 1000 rows:', find(19, '1000'))
print('S 0.14 rows:', find(4, '0.14'))
print('P 0.25 rows:', find(5, '0.25'))
print('C 0.18 rows:', find(1, '0.18'))
print('Si 0.04 rows:', find(2, '0.04'), ' Si 0.07 rows:', find(2, '0.07'))
print('RoA 17 rows:', find(34, '17'))
print('Mn 0.27 rows:', find(3, '0.27'))
print('Martensite 30 rows:', find(42, '30'))
print('FSP 100 rows:', find(40, '100'))
for (col, val) in [(32, '1151'), (35, '188')]:
    for i, wid in find(col, val):
        print(f'col{col}={val} row {i} full:', rows[i])
