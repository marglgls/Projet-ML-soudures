"""Sandbox test: rebuild columns.json metrics from the committed base file
and check the result matches context/columns.json byte-for-byte.
Run from the repo root:  python3 repro_check.py (requires repro/ setup)."""
import json
import shutil
import subprocess
from pathlib import Path

root = Path('.')
repro = root / 'repro'
repro.mkdir(exist_ok=True)
(repro / 'data').mkdir(exist_ok=True)
(repro / 'data' / 'welddb').mkdir(exist_ok=True)
(repro / 'context').mkdir(exist_ok=True)
shutil.copyfile(root / 'data' / 'welddb' / 'welddb.data',
                repro / 'data' / 'welddb' / 'welddb.data')
base = subprocess.run(['git', 'show', 'HEAD:context/columns.json'],
                      capture_output=True, cwd=root).stdout
(repro / 'context' / 'columns.json').write_bytes(base)

for script in ('update_columns.py', 'add_outliers.py'):
    r = subprocess.run(['python3', str(root.resolve() / script)],
                       capture_output=True, text=True, cwd=repro)
    print(script, '->', r.stdout.strip() or r.stderr.strip())

got = (repro / 'context' / 'columns.json').read_bytes()
want = (root / 'context' / 'columns.json').read_bytes()
print('byte-identical to context/columns.json:', got == want)
if got != want:
    g, w = json.loads(got), json.loads(want)
    print('same keys:', set(g) == set(w))
