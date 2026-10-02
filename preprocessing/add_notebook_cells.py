"""Append the '<...'-replacement preprocessing step to data_preprocessing.ipynb."""
import json
import uuid
from pathlib import Path

NB = Path('data_preprocessing.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

# Drop the two trailing empty cells (empty code + empty markdown).
nb['cells'] = [c for c in nb['cells']
               if not (''.join(c.get('source', [])).strip() == '')]


def md(lines):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8],
            'metadata': {}, 'source': lines}


def code(lines, output_text):
    return {'cell_type': 'code', 'execution_count': 7,
            'id': uuid.uuid4().hex[:8], 'metadata': {},
            'outputs': [{'name': 'stdout', 'output_type': 'stream',
                         'text': output_text}],
            'source': lines}


md_cell = md([
    "## Preprocessing step 1 — left-censored `<...` values\n",
    "\n",
    "In the raw file, a token such as `<0.002` means the element was **below the\n",
    "detection limit 0.002** (left-censored), not zero and not missing — see the\n",
    "`censored_tokens` entries in `context/columns.json`. Pandas parses these as\n",
    "strings, so they must be resolved before any numeric modelling.\n",
    "\n",
    "**Rule applied** (`preprocessing/decensor.py`): every `<x` token is replaced\n",
    "by the numeric value `x`, e.g. `<0.002` → `0.002`. This is an **upper-bound\n",
    "choice** (the true value lies in `[0, x]`); the classic alternative is the\n",
    "half-limit `x/2`, or a censored-data model. Revisit if rows at detection\n",
    "limits turn out to drive a model.\n",
    "\n",
    "**Scope**: the 14 numerical columns whose schema documents censored tokens\n",
    "(cols 4, 8–12, 14, 16–21, plus col 39 primary ferrite `<1`, which is an output\n",
    "column). Everything else is out of scope here: `N` stays missing (empty field\n",
    "in the output), and the other encodings need their own steps (col 15\n",
    "`NNtotMMres`, col 27 `150-200` range, col 37 hardness `Hv` suffixes).\n",
    "\n",
    "**Output**: `data/intermediate/decensored.csv` — one row per weld (1652 rows)\n",
    "with the `weld_id` identifier (column 44) and the converted values of the 14\n",
    "concerned columns (`colNN_Label` headers). Plain numerics pass through\n",
    "unchanged; the raw file is never modified.\n",
    "\n",
    "**Caveats to keep in mind**\n",
    "\n",
    "- Replaced values pile up exactly at the limits (e.g. 343 Al rows become 5.0,\n",
    "  395 B rows become 5.0, 266 V rows become 0.0005): expect artificial spikes\n",
    "  in those distributions.\n",
    "- Vanadium `<5` (9 rows → `5.0`) is unit-ambiguous per the schema note on col 9\n",
    "  — flag for review rather than silently trusting it.\n",
    "- `decensored.csv` is git-ignored intermediate data (see `.gitignore`); only the\n",
    "  folder structure is tracked. Re-run the cell below to regenerate it."
])

code_src = [
    "import csv\n",
    "import sys\n",
    "sys.path.insert(0, \"preprocessing\")\n",
    "from decensor import main\n",
    "\n",
    "# Applies '<x' -> x and writes data/intermediate/decensored.csv\n",
    "counts = main()\n",
    "\n",
    "# Round-trip check: 1652 data rows, no '<' token left in value columns\n",
    "rows = list(csv.reader(open(\"data/intermediate/decensored.csv\", encoding=\"utf-8\")))\n",
    "assert len(rows) == 1653\n",
    "assert all(not c.startswith(\"<\") for r in rows[1:] for c in r[1:])\n",
    "print(\"check: 1652 data rows, no '<' token left, header:\", rows[0][:4], \"...\")"
]

code_out = [
    "Concerned columns (14): 4, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 39\n",
    "Replacements '<x' -> x per column: col04=7, col08=2, col09=308, col10=14, col11=21, col12=12, col14=70, col16=403, col17=419, col18=299, col19=5, col20=8, col21=6, col39=2 (total 1576)\n",
    "Wrote data\\intermediate\\decensored.csv: 1652 rows x 15 columns\n",
    "Examples (weld_id | column | raw -> new):\n",
    "  Evans-Al/CMn-1990-<5aw | col09_Vanadium_concentration | <0.0005 -> 0.0005\n",
    "  Evans-Al/CMn-1990-<5aw | col16_Aluminium_concentration | <5 -> 5.0\n",
    "  Evans-Al/CMn-1990-<5aw | col17_Boron_concentration | <5 -> 5.0\n",
    "  Evans-Al/CMn-1990-<5aw | col18_Niobium_concentration | <5 -> 5.0\n",
    "  Evans-Al/CMn-1990-<5awch1 | col09_Vanadium_concentration | <0.0005 -> 0.0005\n",
    "  Evans-Al/CMn-1990-<5awch1 | col16_Aluminium_concentration | <5 -> 5.0\n",
    "  Evans-Al/CMn-1990-<5awch1 | col17_Boron_concentration | <5 -> 5.0\n",
    "  Evans-Al/CMn-1990-<5awch1 | col18_Niobium_concentration | <5 -> 5.0\n",
    "check: 1652 data rows, no '<' token left, header: ['weld_id', 'col04_Sulphur_concentration', 'col08_Molybdenum_concentration', 'col09_Vanadium_concentration'] ...\n"
]

nb['cells'].append(md_cell)
nb['cells'].append(code(code_src, code_out))
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('cells now:', len(nb['cells']))
