"""Preprocessing step 6 — modelling interface on data/cleaned/cleaned.csv.

cleaned.csv is fully numeric/categorical but still has missing values and raw
scales. Imputation, scaling and one-hot encoding *learn* statistics (medians,
means, std, categories), so they must be fitted on the training fold only to
avoid leakage: they are returned as an unfitted sklearn transformer
(`make_preprocessor`) to put in front of any model in a Pipeline, instead of
being applied once to the whole file.

This module also defines:
  - the targets (one model per target, trained on the rows reporting it);
    Charpy energy is temperature-conditioned, so the Charpy test temperature
    (col 35) is added as an input for that target only;
  - the merge of duplicate inputs (see get_xy): rows with exactly the same
    inputs become one row (mean target, article of the first row);
  - the inputs: cols 1-30 kept by build_model_table.py (composition, welding
    parameters, PWHT) + two 0/1 heat-treatment indicators (INDICATORS). Other
    outputs are never used as inputs;
  - `source`: the literature source parsed from the Weld ID, used as CV group
    (same weld as-welded/PWHT/Charpy rows and same experimental series stay
    in the same fold).

Usage (from the repo root):
    import sys; sys.path.insert(0, 'preprocessing')
    from ml_dataset import load_cleaned, get_xy, make_preprocessor
    df = load_cleaned()
    X, y, groups = get_xy(df, 'yield')
    model = Pipeline([('prep', make_preprocessor(X)), ('reg', Ridge())])
    cross_val_score(model, X, y, groups=groups, cv=GroupKFold(5))

Run from the repo root to print a summary:  python3 preprocessing/ml_dataset.py
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path.cwd()
CLEANED = REPO / 'data' / 'cleaned' / 'cleaned.csv'
HEADER_RE = re.compile(r'^col(\d+)_')
LAST_INPUT_COL = 30  # cols 1-30 are inputs, 31+ are outputs
CATEGORICAL = ['col24_AC_or_DC', 'col25_Electrode_polarity', 'col28_Type_of_weld']
CHARPY_T = 'col35_Charpy_temperature'
PWHT_T, PWHT_TIME = 'col29_Post_weld_heat_treatment_temperature', 'col30_Post_weld_heat_treatment_time'
# 0/1 heat-treatment indicators added as inputs. A temperature alone would let a
# model treat 0 (as-welded) and 250 degC (degassing) as points on the same scale
# as a real tempering at 550-750 degC, while they are different kinds of state.
INDICATORS = {
    'as_welded': lambda df: (df[PWHT_T] == 0) & (df[PWHT_TIME] == 0),
    'degassing_250C_14h': lambda df: (df[PWHT_T] == 250) & (df[PWHT_TIME] == 14),
}
TARGETS = {
    'yield': 'col31_Yield_strength',
    'uts': 'col32_Ultimate_tensile_strength',
    'elongation': 'col33_Elongation',
    'roa': 'col34_Reduction_of_Area',
    'charpy': 'col36_Charpy_impact_toughness',
}
# Same source spelled two ways in the Weld IDs (same authors, same year).
SOURCE_ALIAS = {
    'Ga&K-1975': 'Gar&K-1975',
    'KocakPEv-TiN-1994': 'KocakPREv-TiN-1994',
}


def weld_source(weld_id):
    """Literature source of a Weld ID, e.g. 'Evans-AlTi-1994-450/1ch1' ->
    'Evans-AlTi-1994', 'p9-RR82011' -> 'RR82011', 'Mart-Bc' -> 'Mart'."""
    m = re.match(r'^p\d+-(.+)$', weld_id)
    if m:
        return m.group(1)
    if weld_id.startswith('EvansLetter'):
        return 'EvansLetter'
    m = re.match(r'^(.*?19\d\d(?:/19\d\d)?)', weld_id)
    src = m.group(1) if m else weld_id.split('-')[0]
    return SOURCE_ALIAS.get(src, src)


def col_num(header):
    return int(HEADER_RE.match(header).group(1))


def load_cleaned(path=CLEANED):
    """cleaned.csv as a DataFrame (empty field -> NaN) + a `source` column.
    Categorical columns are read as strings (polarity '0' is a code, not 0)."""
    df = pd.read_csv(path, dtype={c: object for c in CATEGORICAL})
    df.insert(1, 'source', df['weld_id'].map(weld_source))
    for name, rule in INDICATORS.items():
        df[name] = rule(df).astype(int)
    return df


def input_columns(df):
    cols = [c for c in df.columns if HEADER_RE.match(c) and col_num(c) <= LAST_INPUT_COL]
    return cols + [c for c in INDICATORS if c in df.columns]


def get_xy(df, target):
    """X, y, groups for one target ('yield', 'uts', 'elongation', 'roa',
    'charpy'), in two steps:

    1. keep the rows where the target is reported;
    2. merge duplicate inputs: rows with exactly the same inputs (same recipe,
       plus the same test temperature for Charpy) are repeated measurements
       of one recipe. They become ONE row: target = mean of the measurements,
       group = article of the first row. Two empty cells count as equal.

    Called on the whole base, before the CV split. No leakage: each mean only
    uses the measurements of its own recipe, and the merged row then goes
    entirely to train or to test (without the merge, two replicates could be
    split between train and test).
    """
    y_col = TARGETS[target]
    rows = df[df[y_col].notna()]
    x_cols = input_columns(df) + ([CHARPY_T] if target == 'charpy' else [])
    merged = (rows.groupby(x_cols, dropna=False, sort=False)
                  .agg({y_col: 'mean', 'source': 'first'})
                  .reset_index())
    return merged[x_cols], merged[y_col], merged['source']


def make_preprocessor(X, impute='median'):
    """Unfitted ColumnTransformer for the columns of X.

    Numeric: impute (`impute` = 'median' or 'mean', fitted on the training
    fold) + one 0/1 'was missing' indicator per column with missing values
    (missingness is structural: it depends on the source paper), then
    standardise. Categorical: missing -> its own 'missing' level, then
    one-hot (levels unseen in training are ignored at test time).
    """
    categorical = [c for c in X.columns if c in CATEGORICAL]
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy=impute, add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', num_pipe, numeric),
        ('cat', cat_pipe, categorical),
    ], verbose_feature_names_out=False)


def main():
    df = load_cleaned()
    print(f'Loaded {CLEANED.relative_to(REPO)}: {df.shape[0]} rows, '
          f'{len(input_columns(df))} input columns, {df["source"].nunique()} sources')
    for target, y_col in TARGETS.items():
        X, y, groups = get_xy(df, target)
        Xt = make_preprocessor(X).fit_transform(X)
        print(f'  {target:<10} {y_col:<34} n={len(y):>4} (from {int(df[y_col].notna().sum()):>4} rows)  sources={groups.nunique():>2}  '
              f'X after preprocessing: {Xt.shape[1]} features, NaN left: {int(pd.isna(Xt).sum())}')


if __name__ == '__main__':
    main()
