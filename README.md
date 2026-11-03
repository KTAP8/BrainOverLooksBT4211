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
