"""Check that context/columns.json differs from the committed HEAD version
only by added metric keys (mean/median/n_numeric, categorical stats,
numeric_range stats, outliers) — no other value was altered.
Run from the repo root:  python3 verify_out.py
"""
import copy
import json
import subprocess

old = json.loads(subprocess.run(['git', 'show', 'HEAD:context/columns.json'],
                  capture_output=True).stdout.decode('utf-8'))
new = json.load(open('context/columns.json', encoding='utf-8'))
ADDED_TOP = {'mean', 'median', 'n_numeric', 'n_categories',
             'category_count_stats', 'outliers'}
ADDED_NR = {'mean', 'median', 'n_numeric'}


def strip(obs):
    obs = copy.deepcopy(obs)
    for k in list(obs):
        if k in ADDED_TOP:
            del obs[k]
    nr = obs.get('numeric_range')
    if isinstance(nr, dict):
        for k in list(nr):
            if k in ADDED_NR:
                del nr[k]
    return obs


ok = True
for k in old:
    if strip(old[k]['observed']) != strip(new[k]['observed']):
        print('SEMANTIC CHANGE in col', k)
        ok = False
    for f in ('label', 'unit', 'details', 'type', 'role', 'missing'):
        if old[k].get(f) != new[k].get(f):
            print('HEADER CHANGE in col', k, f)
            ok = False
print('only-additions:', ok)
