# Projet-ML-soudures
Projet Machine Learning 3A CentraleSupélec sur la prédiction de qualité de soudures

## Dataset context — MAP_DATA_WELD (`data/welddb/`)

All-weld-metal deposit database by Tracey Cool and H. K. D. H. Bhadeshia (Phase Transformations Group, Dept. of Materials Science and Metallurgy, University of Cambridge), collated from the literature for neural-network modelling of steel weld deposits.

- References: T. Cool, *Design of Steel Weld Deposits*, PhD thesis, Univ. of Cambridge, 1996; Cool, Bhadeshia & MacKay, *Materials Science and Engineering*, 1997.
- Files: `welddb.data` (data, no header), `welddb.info` / `welddb.tex` (column documentation, same content), `context/data_explantions.md` (copy of the column list). Machine-readable column reference: `context/columns.json` (enriched with observed stats from the `.data` file).
- Shape: **1652 rows × 44 columns**, whitespace-separated, no header. Column order = Column 1…44 as documented. **Missing = `N`** (not reported, not zero).
- `Weld ID` (col 44): 1490 unique IDs for 1652 rows; 91 IDs appear 2–5× (e.g. same weld as-welded + PWHT, or retested at another Charpy temperature). Split by ID/source, not by row.

### Column groups

- **Composition inputs (1–21):** C, Si, Mn, S, P, Ni, Cr, Mo, V, Cu, Co, W in wt% (1–12); O, Ti, N, Al, B, Nb, Sn, As, Sb in ppmw (13–21). Cols 1–5 are complete; alloy/trace elements are sparse (Co 92% missing, W 96%, Sn/As/Sb 82–86%).
- **Welding parameters (22–28):** Current (A), Voltage (V), AC/DC, electrode polarity (`+`/`-`/`0`), heat input (kJ/mm), interpass T (°C), weld process. `0` polarity = AC with no polarity (38 rows, AC-only). `150-200` interpass token (38 rows) is a range. Process codes in file: MMA (1140), SA (261), FCA (87), TSA (87), ShMA (40), NGSAW (18, = SAW-NG), NGGMA (7, = GMA-NG), SAA (4, undocumented SA variant), GTAA/GMAA (4 each). MMA and ShMA both = manual metal arc.
- **PWHT (29–30):** temperature (°C) / time (h). `0/0` = as-welded (610 rows). 13 rows missing both.
- **Mechanical outputs (31–38):** yield strength, UTS, elongation, reduction of area, Charpy test T (condition) + Charpy energy, hardness, 50% FATT. Best-supported targets: Charpy energy (879), yield (780), UTS (738), elongation (700), RoA (705). Hardness (138 rows, 8%) mixes Vickers loads (`201Hv10`, `143(Hv30)`, `226Hv5`); FATT has only 31 values.
- **Microstructure outputs (39–43, %):** primary ferrite, ferrite with second phase, acicular ferrite, martensite (88× `0.0` + 1× `30.0`), ferrite with carbide aggregate. Only ~89–98 rows each.

### Caveats for modelling

1. **Missingness is structural** (unreported in source papers), not MCAR — complete-case analysis discards most rows; prefer per-target subsets or imputation.
2. **Non-numeric tokens need parsing:** `<...` detection limits (cols 4, 8–12, 14, 16–21, 39), `150-200` (col 27), `NNtotMMres` nitrogen encoding with `nd` (col 15, 59 rows, e.g. `67tot33res`), hardness load suffixes (col 37).
3. **Charpy energy is temperature-conditioned** — always model cols 35+36 jointly.
4. **Duplicate/grouped IDs** — keep same-ID rows in the same fold; Weld ID encodes the literature source (Evans 700 rows, etc.), so group-by-source splits test generalisation best.
5. **Class imbalance** in process (MMA-dominated) and current type (DC 1395 vs AC 42).

## Preprocessing pipeline

Data is git-ignored. Run `data_preprocessing.ipynb` top to bottom (or the scripts below, from the repo root). It will start by downloading https://www.phase-trans.msm.cam.ac.uk/map/data/tar/welddb.tar and extracting `welddb.data` into `data/welddb/`. Every decision is justified in the notebook.

1. `preprocessing/decensor.py` — `<x` detection limits → `x` (unit fixes: V `<5` = 5 ppmw; Ti, Al `<0.01` = 0.01 wt% = 100 ppmw).
2. `preprocessing/nitrogen_total.py` — `NNtotMMres` → total N `NN`.
3. `preprocessing/interpass_range.py` — `150-200` → `175`.
4. `preprocessing/categorical_clean.py` — AC/DC filled from polarity; weld process grouped into MMA, SA, TSA, FCA, GSA.
4b. `preprocessing/heat_treatment.py` — 13 unknown heat treatments (`Gar&K-1975-25mm-*ht`) set to the same article's 600 °C / 0.75 h (assumption).
5. `preprocessing/build_model_table.py` — joins `data/intermediate/*.csv` into `data/cleaned/cleaned.csv` (1652 × 30, sparse columns dropped).
5b. `preprocessing/alloy_zero.py` (called by step 5) — Ni, Cr, Mo, V, Nb never reported by an article (empty on all its rows) → 0: a reporting choice meaning "not alloyed", not a missing measurement. Applied article by article, before any merge of articles. Isolated gaps, O, N, Ti, Al and welding parameters stay missing (median imputation in each CV fold).
6. `preprocessing/ml_dataset.py` — modelling interface: `get_xy(df, target)` for `yield`, `uts`, `elongation`, `roa`, `charpy` (+ Charpy temperature as input), `source` CV groups, heat-treatment indicators (`as_welded`, `degassing_250C_14h`), and `make_preprocessor(X)` (median imputation + missing indicators, scaling, one-hot) to fit inside each CV fold:

```python
import sys; sys.path.insert(0, 'preprocessing')
from ml_dataset import load_cleaned, get_xy, make_preprocessor
X, y, groups = get_xy(load_cleaned(), 'yield')
model = Pipeline([('prep', make_preprocessor(X)), ('reg', Ridge())])
cross_val_score(model, X, y, groups=groups, cv=GroupKFold(5))
```
