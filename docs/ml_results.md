# Part 4: machine-learning cross-check

Owner: Tommy. Updated 2026-11-03. Depends on Part 3 (`docs/core_model_results.md`).
This section tries to break the core model and reports that it could not.

## Headline

A gradient booster, given the same six ratings and free to use non-linearity and
interactions, **loses to logistic regression in all six configurations** (AUC
margin −0.006 to −0.023). There is **no dealbreaker threshold** and **no useful
interaction**. Part 3's linear, additive specification is the right one.

The one place the methods genuinely diverge is not the ranking but the
coefficients: **dropping person fixed effects attenuates all twelve estimates**,
four of them below zero.

## Results

### 1. Does a flexible model predict better?

Grouped 5-fold CV by participant; `person-demeaned` subtracts each person's own
mean rating first.

| Sample | Features | Rows | Logistic AUC | GBM AUC | GBM − logistic |
|---|---|---:|---:|---:|---:|
| Women | standardised | 3,000 | 0.8153 | 0.7969 | −0.0185 |
| Women | person-demeaned | 3,000 | 0.8344 | 0.8115 | −0.0229 |
| Men | standardised | 3,222 | 0.8353 | 0.8294 | −0.0060 |
| Men | person-demeaned | 3,222 | 0.8365 | 0.8292 | −0.0073 |
| Pooled | standardised | 6,222 | 0.8221 | 0.8151 | −0.0070 |
| Pooled | person-demeaned | 6,222 | 0.8336 | 0.8241 | −0.0094 |

Demeaning improves both models, which is the fixed-effects argument restated in
feature space: how a partner compares with that person's own average predicts
better than the absolute rating.

### 2. Three importance estimates

Project convention throughout: `100 * max(coef, 0) / sum of positives`.

| Gender | Attribute | Cond. logit | Pooled logit | GBM perm. | Cond. coef | Pooled coef | Shift |
|---|---|---:|---:|---:|---:|---:|---:|
| Women | Attractive | 37.86 | 39.46 | 43.38 | +1.60 | +0.86 | −0.74 |
| Women | Sincere | 9.18 | 0.00 | 6.66 | +0.39 | **−0.22** | −0.61 |
| Women | Intelligent | 9.37 | 6.16 | 1.87 | +0.40 | +0.13 | −0.26 |
| Women | Fun | 19.80 | 26.41 | 18.93 | +0.84 | +0.58 | −0.26 |
| Women | Ambitious | 1.38 | 0.00 | 3.11 | +0.06 | **−0.25** | −0.30 |
| Women | Shared interests | 22.41 | 27.98 | 26.06 | +0.95 | +0.61 | −0.34 |
| Men | Attractive | 47.37 | 53.35 | 66.60 | +1.90 | +1.25 | −0.65 |
| Men | Sincere | 2.19 | 0.00 | 1.93 | +0.09 | **−0.21** | −0.30 |
| Men | Intelligent | 6.03 | 0.00 | 2.87 | +0.24 | **−0.04** | −0.28 |
| Men | Fun | 24.03 | 24.34 | 13.32 | +0.97 | +0.57 | −0.40 |
| Men | Ambitious | 0.00 | 0.00 | 0.95 | −0.06 | **−0.27** | −0.21 |
| Men | Shared interests | 20.39 | 22.31 | 14.34 | +0.82 | +0.52 | −0.30 |

Spearman rank agreement:

| Gender | GBM vs pooled | GBM vs conditional | Pooled vs conditional |
|---|---:|---:|---:|
| Women | 0.812 | 0.829 | 0.986 |
| Men | 0.880 | 0.943 | 0.941 |

**The ranking is robust; the estimates are not.** All three methods put
attractiveness first and ambition last. But every one of the twelve coefficients
moves toward zero when the person effect is dropped (−0.21 to −0.74 SD), and four
cross below zero. Because the agreed importance rule clips negatives, the pooled
model reports those four as exactly `0.00`, which reads as "no effect" when it
means "estimated negative". The fixed effects are load-bearing, and clipped
shares alone hide that — so the signed coefficients belong in the appendix.

### 3. Threshold and interactions

| Attractiveness rating | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P(yes) | .056 | .064 | .062 | .113 | .208 | .357 | .559 | .747 | .790 | .847 |

Smooth and monotone, and the GBM's out-of-fold fitted curve lies on top of it.

- Ten rating dummies vs one linear term: log-loss **0.54653 vs 0.54753**, a gain
  of **0.00088**.
- All 15 pairwise interactions added to the logistic: AUC **0.82195 vs 0.82207**,
  a change of **−0.00012**.

Preferences are **compensatory**: no attribute acts as a gate. That answers
sub-question 2 in `PROJECT_CONTEXT.md` with a clean negative.

## Method

- **Sample:** the union of Part 3's two retained samples — 6,222 decisions from
  452 people. `prepare_ml_frame` re-derives it and raises if it does not match
  `outputs/tables/core_model_sample_flow.csv`, so this section cannot silently
  analyse different data from Part 3.
- **Standardisation:** Part 3's own within-gender means and SDs, read from
  `outputs/tables/core_model_scaling.csv`. A pooled SD would make these
  coefficients incomparable with the published ones.
- **Predictors:** only `attr`, `sinc`, `intel`, `fun`, `amb`, `shar`. `match`,
  `dec_o`, `like`, `prob` are excluded, as are all `_o` partner-side ratings.
- **Cross-validation:** `GroupKFold(5)` on `iid`. A held-out person never appears
  in training; a row-level split would leak their selectivity.
- **Booster:** scikit-learn `HistGradientBoostingClassifier` (300 iterations,
  learning rate 0.06, 15 leaves, min 40 per leaf, L2 1.0, `SEED = 42`).
  `PROJECT_CONTEXT.md` names XGBoost or LightGBM; both require an OpenMP runtime
  (`libomp`) that is not present on all our machines, and scikit-learn is already
  a dependency. Swap `make_gbm` if you have XGBoost installed.
- **Permutation importance:** measured out of fold, 10 repeats, averaged over the
  5 folds, scored on AUC.

## Validation

Nine tests in `tests/test_ml_model.py`, each able to fail for a real reason: the
sample matches Part 3's published flow; every retained person has both outcomes;
`z_` columns reproduce Part 3's scaler and are mean-0/SD-1 within gender; `d_`
columns are person-centred; the sample assertion actually fires on altered data;
no person spans a train/test split; the booster is deterministic; the seed is the
project seed. The notebook runs top to bottom from a fresh kernel and retains its
outputs. Environment and diagnostics are in `outputs/tables/ml_diagnostics.json`.

## Interpretation and limitations

The managerial point is that the simple model survived a deliberate attempt to
beat it, which is what makes Part 3's numbers worth presenting. "We used a more
flexible model and it did not help" is evidence, not a failure.

Permutation importance and a clipped coefficient share are **different
quantities**; the three columns are comparable in rank, not in level. AUC gaps of
0.01 sit inside fold-to-fold noise (fold SDs 0.003–0.037), so read these as "the
booster does not win", not "the logit is better". The booster was not tuned
beyond sensible defaults — a heavily tuned one might close the gap, though it
would still face the same compensatory, threshold-free structure. Twelve of the
ratings are non-integer and are binned to the nearest whole number for the curve
only. Everything here is the same observational data as Part 3, so none of it is
causal.

## Files and rerunning

- `notebooks/03_ml.ipynb` — executed analysis.
- `src/ml_model.py` — implementation; `python -m src.ml_model` from the repo root.
- `outputs/figures/ml_fixed_effect_shift.png` — the attenuation result.
- `outputs/figures/ml_importance_comparison.png` — three methods, one ranking.
- `outputs/figures/ml_attractiveness_curve.png` — no threshold.
- `outputs/tables/ml_benchmark.csv`, `ml_importance.csv`, `ml_rank_agreement.csv`,
  `ml_attractiveness_curve.csv`, `ml_diagnostics.json`.
- Tests: `python -m unittest discover -s tests -v`.

## Sources and AI use

- [Part 3 results](core_model_results.md) and [project specification](../PROJECT_CONTEXT.md).
- Fisman, Iyengar, Kamenica & Simonson (2006), *Gender Differences in Mate
  Selection: Evidence From a Speed Dating Experiment*, QJE 121(2), 673–697.
  DOI [10.1162/qjec.2006.121.2.673](https://doi.org/10.1162/qjec.2006.121.2.673) (resolves to Oxford Academic).
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
  and [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
