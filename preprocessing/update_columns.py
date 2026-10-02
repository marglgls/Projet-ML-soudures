"""Mean / median (+ n_numeric basis) for numeric columns, n_categories +
category_count_stats for categorical columns. Updates context/columns.json.
Run from the repo root:  python3 update_columns.py
"""
import json
import statistics

rows = [l.split() for l in open('data/welddb/welddb.data', encoding='utf-8') if l.strip()]
cols = json.load(open('context/columns.json', encoding='utf-8'))


def numvals(j):
    """Float-parseable values of column j, excluding missing 'N' and
    non-numeric tokens ('<...', '150-200', 'NNtotMMres', 'Hv' suffixes)."""
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


def r4(x):
    return round(x, 4) if isinstance(x, float) else x


for k, entry in cols.items():
    t = entry['type']
    obs = entry['observed']
    j = int(k) - 1
    if t.startswith('continuous'):
        v = numvals(j)
        mean, med = r4(statistics.mean(v)), r4(statistics.median(v))
        if 'numeric_range' in obs:
            # quirk columns (15, 27, 37): stats describe the plain-numeric subset
            nr = obs['numeric_range']
            new_nr = {}
            for kk, vv in nr.items():
                new_nr[kk] = vv
                if kk == 'max':
                    if 'plain_numeric' not in obs:
                        new_nr['n_numeric'] = len(v)
                    new_nr['mean'] = mean
                    new_nr['median'] = med
            obs['numeric_range'] = new_nr
        elif 'max' in obs:
            new_obs = {}
            for kk, vv in obs.items():
                new_obs[kk] = vv
                if kk == 'max':
                    if 'censored_tokens' in obs:
                        new_obs['n_numeric'] = len(v)
                    new_obs['mean'] = mean
                    new_obs['median'] = med
            entry['observed'] = new_obs
        else:
            # continuous without min/max (col 42 martensite): append after n_reported
            new_obs = {}
            for kk, vv in obs.items():
                new_obs[kk] = vv
                if kk == 'n_reported':
                    new_obs['mean'] = mean
                    new_obs['median'] = med
            entry['observed'] = new_obs
    elif t == 'categorical':
        counts = list(obs['categories'].values())
        new_obs = {}
        for kk, vv in obs.items():
            new_obs[kk] = vv
            if kk == 'categories':
                new_obs['n_categories'] = len(counts)
                new_obs['category_count_stats'] = {
                    'min': min(counts),
                    'max': max(counts),
                    'mean': r4(statistics.mean(counts)),
                    'median': r4(statistics.median(counts)),
                }
        entry['observed'] = new_obs

with open('context/columns.json', 'w', encoding='utf-8') as f:
    json.dump(cols, f, indent=2, ensure_ascii=False)
    f.write('\n')
print('done')
