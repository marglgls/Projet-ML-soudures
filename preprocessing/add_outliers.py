"""Annotate context/columns.json with the reviewed outlier findings
(counts, values, row/source notes). Fails if 'outliers' already present.
Run from the repo root, AFTER update_columns.py:  python3 add_outliers.py
"""
import json

cols = json.load(open('context/columns.json', encoding='utf-8'))

OUT = {
    "1": {"high": {"count": 1, "values": [0.18],
        "note": "Single max 0.18 is Birmingham-MAX39, the documented 9Cr-type special weld (see Cr note) — explained, not a typo."}},
    "2": {"low": {"count": 2, "values": [0.04, 0.07],
        "note": "Two isolated low singles (Wolst-1974-AS0/AS1); 0.04 shares its row with the O maximum — old-source reporting level. High tail to 1.14 is stepped (mostly pairs), a systematic high-Si series, not outliers."}},
    "3": {"low": {"count": 1, "values": [0.27],
        "note": "Single min 0.27 (Wolst-1974-AM1, next value 0.44); same 1974 source holds other dataset extremes (see O, Si, RoA notes)."}},
    "4": {"high": {"count": 10, "values": [0.14],
        "note": "Plateau of 10 rows at 0.14 (Mart-G..Gi series, gap from next value 0.036); same 10 rows are the P maximum — deliberate high-impurity series. Keep together; do not drop as errors."}},
    "5": {"high": {"count": 10, "values": [0.25],
        "note": "Same 10 Mart- rows as the S plateau (gap from next value 0.075). Deliberate series."}},
    "11": {"high": {"count": 5, "values": [0.92, 0.95, 1.87, 1.89, 2.8],
        "note": "Five isolated high singles vs bulk <=0.021 (n=108, mostly 0) — distinct high-Co alloy additions, not noise. Tiny group: keep in same fold."}},
    "12": {"high": {"count": 6, "values": [0.97, 0.98, 1.03, 1.92, 2.99],
        "note": "Only 6 nonzero of 63 reported (57 zeros) — tungsten-alloyed additions, a separate cluster, not outliers."}},
    "13": {"high": {"count": 8, "values": [1020.0, 1070.0, 1100.0, 1150.0, 1180.0, 1300.0, 1440.0, 1650.0],
        "note": "Isolated high singles above the 922 cluster (each x1); max 1650 (Wolst-1974-AS0) shares its row with the Si minimum. Low tail 132-218 (30 rows) is sparse but continuous — no flag."}},
    "19": {"high": {"count": 2, "values": [1000.0],
        "note": "Two rows at 1000.0 (5x above next value 200) are the same weld as-welded + heat-treated (Cunh-1982-7016aw3/ht3) — deliberate high-Sn pair, keep together; check unit consistency before transforms."}},
    "31": {"high": {"count": 1, "values": [920.0],
        "note": "Single max 920 (gap 86 above 834) shares its row with the UTS max and elongation min (Evans-AlEl9Cr1Mo-1994-3Ni, 9Cr-3Ni weld) — internally consistent ultra-high-strength result, not a typo."}},
    "32": {"high": {"count": 1, "values": [1151.0],
        "note": "Single max 1151 (gap 242 above 909), same 9Cr-3Ni row as the YS max — consistent, not a typo; check leverage."}},
    "33": {"low": {"count": 2, "values": [10.6, 11.0],
        "note": "Minimum 10.6 sits on the same ultra-high-strength row (consistent: lowest ductility at highest strength). Low tail, not errors."}},
    "34": {"low": {"count": 1, "values": [17.0],
        "note": "Single min 17.0 (Wolst-1974-N, gap 7 below next 24) — same 1974 source as other extremes; verify, do not silently drop."}},
    "35": {"high": {"count": 1, "values": [188.0],
        "note": "Single max 188 (gap 118 above next 70; Ditt-0.5rch2) with paired energy 141 J — plausible high-T test but isolated; verify (possible typo) and check leverage in temperature-conditioned models."}},
    "36": {"high": {"count": 1, "values": [270.0],
        "note": "Single max 270 (gap 32 above 238; Chandel&-1985W3ch1). Mild — check leverage only."}},
    "42": {"high": {"count": 1, "values": [30.0],
        "note": "Single nonzero among 89 reported (88 zeros; Evans-Ni/CMn-1990/1991-3.5Dawch) — the only observed martensite; verify, it carries any martensite signal alone."}},
}

for k, o in OUT.items():
    obs = cols[k]['observed']
    assert 'outliers' not in obs, k
    obs['outliers'] = o

with open('context/columns.json', 'w', encoding='utf-8') as f:
    json.dump(cols, f, indent=2, ensure_ascii=False)
    f.write('\n')
print('annotated cols:', sorted(OUT))
