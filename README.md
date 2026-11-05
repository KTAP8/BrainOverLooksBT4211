# People Say They Want Brains, Then Choose Looks

Marketing analytics group project: **stated vs revealed** attribute preferences using the Columbia Speed Dating Experiment (Fisman et al., 2006).

One-line pitch: *Customers tell you what they value. Their choices tell you the truth. How big is the gap, and what should marketers do about it?*

Full framing, methods, and pitfalls: see [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw file at:

```
data/raw/Speed_Dating_Data.csv
```

Do **not** modify the raw CSV. Load with `encoding="cp1252"`.

## Notebook order

| # | Notebook | Owner role | Depends on |
|---|----------|------------|------------|
| 0 | `notebooks/00_audit_cleaning.ipynb` | Any | raw + processed CSVs |
| 1 | `notebooks/01_cleaning.ipynb` | Data lead | raw CSV |
| 2 | `notebooks/02_core_model.ipynb` | Core model | `data/processed/clean_unscaled.csv` |
| 3 | `notebooks/03_ml.ipynb` | ML model | clean_unscaled.csv |
| 4 | `notebooks/04_robustness.ipynb` | Robustness | clean_unscaled.csv |
| 5 | `notebooks/05_customer_value.ipynb` | Implications | clean_unscaled.csv + model outputs |

Run `01_cleaning.ipynb` (or `python -m src.cleaning`) after cloning if `clean_unscaled.csv` is not in the repo.

Run each notebook **top to bottom from a fresh kernel**. Prefer starting Jupyter from the repo root so relative paths resolve.

## Part 3: core model

Open [`notebooks/02_core_model.ipynb`](notebooks/02_core_model.ipynb) to view the executed results or rerun from a fresh kernel. It works from the repository root or `notebooks/`. The same analysis can be run from the root with:

```bash
python -m src.core_model
python -m unittest discover -s tests -v
```

The main chart is [`outputs/figures/core_model_stated_vs_revealed.png`](outputs/figures/core_model_stated_vs_revealed.png). See the [results and method](docs/core_model_results.md) and [presentation notes](docs/core_model_presentation_notes.md). Core-model tables use the `core_model_` prefix. The first run compiles the exact likelihood; 1,000 person-bootstrap draws per gender take a few minutes. Runtime versions are recorded in the diagnostics JSON.

## Part 4: ML cross-check

Open [`notebooks/03_ml.ipynb`](notebooks/03_ml.ipynb) to view the executed results, or rerun from the repository root:

```bash
python -m src.ml_model
python -m unittest discover -s tests -v
```

This section tries to break Part 3 and reports that it could not. A gradient booster given the same six ratings, and free to use thresholds and interactions, **loses in all six configurations** (AUC margin −0.006 to −0.023); there is no dealbreaker cut-off and no useful interaction, so preferences here are compensatory. What does move is the coefficients: dropping the person fixed effects attenuates all twelve estimates, four of them below zero, which the agreed clip-at-0 importance rule then reports as `0.00`.

It runs on the union of Part 3's two retained samples (6,222 decisions, 452 participants) and asserts both the sample and the standardisation against `core_model_sample_flow.csv` and `core_model_scaling.csv`, so this section cannot silently analyse different data from Part 3. ML tables use the `ml_` prefix.

**Deviation from [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) §5:** the booster is scikit-learn's `HistGradientBoostingClassifier` and importance is permutation importance, not XGBoost and SHAP. Both boosters need an OpenMP runtime (`libomp`) that would not load on the machine this was built on, and scikit-learn is already a dependency. `requirements.txt` is unchanged; replacing `make_gbm` is the only edit needed if you have XGBoost working.

See the [results and method](docs/ml_results.md). Main figures: [`ml_fixed_effect_shift.png`](outputs/figures/ml_fixed_effect_shift.png), [`ml_importance_comparison.png`](outputs/figures/ml_importance_comparison.png), [`ml_attractiveness_curve.png`](outputs/figures/ml_attractiveness_curve.png).

## Shared code

Constants, loaders, and plotting helpers live in `src/`:

- `ATTRS`, `STATED`, `LEAKY`, `SEED = 42`
- Fixed stated/revealed chart colours
- `load_raw()` / `load_clean()`

## Outputs

- Figures (300 dpi): `outputs/figures/`
- Result tables: `outputs/tables/`

## AI declaration

Log all AI use in [`AI_USAGE.md`](AI_USAGE.md).
