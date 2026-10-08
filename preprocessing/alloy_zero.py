"""Preprocessing step 5b — alloying elements never reported by an article -> 0.

Ni, Cr, Mo, V (wt%) and Nb (ppmw) are deliberate alloying additions. When an
article never reports one of them (empty on ALL its rows), it is a reporting
choice of the article, not a forgotten measurement: its welds are not alloyed
with that element (residual level, ~0). Evidence from the data: C-Mn welds
that do report these elements give residual values (Ni and Cr median 0 %,
90th percentile <= 0.08 %), far below the medians the imputer would
otherwise use (Cr 0.53 %, Mo 0.34 %): imputing them would invent an alloyed
steel.
Such cells are set to 0 in data/cleaned/cleaned.csv, with no 'was missing'
indicator (team decision). The rule is applied article by article, on the
article parsed from the Weld ID (weld_source, spelling aliases merged) BEFORE
any merge of articles sharing a weld.

Not concerned:
  - isolated missing values (the article reports the element on other rows):
    they stay missing and are imputed by the median inside each CV fold;
  - O, N (always present in weld metal), Ti, Al (deoxidisers) and the
    welding parameters: an absence there does not mean zero.

Called by build_model_table.py just before cleaned.csv is written.
"""
from collections import Counter, defaultdict

from ml_dataset import weld_source

ALLOY_COLUMNS = [
    'col06_Nickel_concentration',
    'col07_Chromium_concentration',
    'col08_Molybdenum_concentration',
    'col09_Vanadium_concentration',
    'col18_Niobium_concentration',
]
ZERO = '0'


def unreported_by_article(headers, rows):
    """Set of (article, column) where the column is empty on every row of the
    article. `rows` are lists of strings, weld_id first, '' = missing."""
    idx = {h: headers.index(h) for h in ALLOY_COLUMNS}
    articles, reported = set(), defaultdict(set)
    for row in rows:
        article = weld_source(row[0])
        articles.add(article)
        for h, i in idx.items():
            if row[i] != '':
                reported[article].add(h)
    return {(a, h) for a in articles for h in ALLOY_COLUMNS if h not in reported[a]}


def fill_zero(headers, rows):
    """Set to 0 the alloy cells of the articles that never report the element.
    Modifies `rows` in place; returns Counter {(article, column): cells set}."""
    unreported = unreported_by_article(headers, rows)
    idx = {h: headers.index(h) for h in ALLOY_COLUMNS}
    filled = Counter()
    for row in rows:
        article = weld_source(row[0])
        for h, i in idx.items():
            if (article, h) in unreported:
                assert row[i] == '', f'{row[0]} {h}: expected empty, got {row[i]!r}'
                row[i] = ZERO
                filled[(article, h)] += 1
    return filled


def report(filled):
    """Print the number of cells set to 0, per element then per article."""
    total = sum(filled.values())
    print(f'Step 5b - alloying elements never reported by an article -> 0: {total} cells')
    for h in ALLOY_COLUMNS:
        arts = sorted(a for (a, c) in filled if c == h)
        n = sum(v for (a, c), v in filled.items() if c == h)
        print(f'  {h}: {n} cells, {len(arts)} articles')
    print('  per article: ' + '; '.join(
        f"{a} [{', '.join(c.split('_')[1] for (aa, c) in sorted(filled) if aa == a)}]"
        for a in sorted({a for a, _ in filled})))
