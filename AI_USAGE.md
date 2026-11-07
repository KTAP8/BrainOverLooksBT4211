# AI usage log

Required for submission. Every team member adds a row when AI is used for code, analysis, or slide text.

| Date | Member | Tool | Used for | What we changed / verified |
| ---- | ------ | ---- | -------- | -------------------------- |
| 2026-11-02 | — | Cursor (Composer) | Scaffolded repo structure from PROJECT_CONTEXT.md; reviewed problem framing | Pending team review of structure |
| 2026-11-02 | — | Cursor (Composer) | Cleaning audit + warning fixes: src/cleaning.py, 01_cleaning, audit updates | Team verified audit CSV (0 warnings) |
| 2026-11-03 | Person 6 | Cursor (Composer) | Part 6: customer_value module, 05 notebook, slide/script docs, README | Verify lift numbers against bootstrap CSV before presenting |
| 2026-11-02 | Jiayi | Codex | Standalone conditional-logit draft, bootstrap analysis, charts and explanation | Codex checked likelihood and fits against statsmodels; standalone draft superseded by repository-aligned analysis below. |
| 2026-11-03 | Jiayi | Codex | Implemented Part 3 to match PROJECT_CONTEXT.md, standardized/clipped importance, participant bootstrap, notebook, charts and speaking notes | Codex ran the notebook from a fresh kernel; all 2,000 draws converged; three numerical tests passed; charts visually checked. Team review pending. |
| 2026-11-06 | Tommy | Claude Code | Implemented Part 4: src/ml_model.py, 03_ml notebook, tests/test_ml_model.py, three figures and docs/ml_results.md | Sample and standardisation are asserted in code against Part 3's published tables; 9 tests passed and the notebook ran 10/10 cells from a fresh kernel (pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1, recorded in ml_diagnostics.json). An earlier claim about ambition was retracted after tracing it to an inconsistent importance convention in a scratch script. Booster and importance method deviate from §5 (see README). Team review pending. |
| 2026-11-07 | TBC | Claude Code | Extended analyses: src/ext_*.py, src/extended_analysis.py, 06_extended_analysis notebook, tests/test_extended.py, ext_ tables and figures, docs/extended_analysis_report.md, README section | 30 tests passed; two full runs produced identical tables; notebook ran from a fresh kernel from the repo root and from notebooks/; sample asserted against Part 3; report numbers checked against the CSVs; citations checked online. Early leakage bugs and overclaims were corrected and listed in the report. Team review pending. |
