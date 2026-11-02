# Cleaning audit verdict

Generated from `notebooks/00_audit_cleaning.ipynb` / `src/audit_cleaning.py`. Full checklist: `outputs/tables/cleaning_audit.csv`.

## Summary (after warning fixes)

**All checks passed — 0 warnings, 0 failures.**

| Status | Count |
|--------|-------|
| pass | 40 |
| warn | 0 |
| fail | 0 |

## Canonical file

Use only [`data/processed/clean_unscaled.csv`](../data/processed/clean_unscaled.csv).

The legacy z-scored `clean.csv` has been **removed**. Standardise partner ratings inside model notebooks (`02`, `03`), not in the shared clean file.

## Pipeline

- Logic: [`src/cleaning.py`](../src/cleaning.py) (`run_cleaning` / `clean_dataframe`)
- CLI: `python -m src.cleaning` or [`notebooks/01_cleaning.ipynb`](../notebooks/01_cleaning.ipynb)
- Preference rescale includes `*1_1`, `*4_1`, `*2_1`, `*1_2`, `*7_2`, and `pf_o_*`

## Documented policies (01_cleaning)

- Zero and out-of-range 1–10 ratings → missing; no imputation
- ~121 rows (1.4%) lack complete `*1_1` sextets — left missing, not dropped
- 8,378 rows preserved throughout
