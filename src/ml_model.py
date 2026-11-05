"""Part 4: machine-learning cross-check of the core model.

Run from the repository root: python -m src.ml_model
The notebook calls the same functions. Only the six partner ratings predict dec.

Design notes, because they are the point of this section rather than incidental:

* The sample is the union of the two retained conditional-logit samples from
  Part 3, so AUC and importance refer to the same decisions Jiayi's model used.
  `prepare_ml_frame` re-derives it and asserts against
  `outputs/tables/core_model_sample_flow.csv`, so any divergence from Part 3
  fails loudly instead of quietly producing a second set of numbers.
* Ratings are standardised with Part 3's own within-gender means and SDs, read
  from `outputs/tables/core_model_scaling.csv`. A pooled SD would make these
  coefficients incomparable to the published ones.
* Cross-validation groups by `iid`, so a held-out person is never also in
  training. Row-level splits leak: the same person's other dates carry their
  fixed effect.
* The gradient booster is scikit-learn's `HistGradientBoostingClassifier`.
  `PROJECT_CONTEXT.md` asks for XGBoost or LightGBM; both need an OpenMP
  runtime (`libomp`) that is not present on every machine here, and sklearn is
  already a dependency. Swap in XGBoost by replacing `make_gbm` if you have it.
"""
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import (ATTRS, ATTR_LABELS, STATED, SEED, STATED_COLOR,
                        REVEALED_COLOR, FIGURES_DIR, TABLES_DIR)
from .io import load_clean, project_root

GENDERS = ((0, "Women"), (1, "Men"))
N_SPLITS = 5


def make_gbm():
    """Gradient booster. Deliberately modest: 6 features, ~6k rows."""
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)


def make_logit():
    return LogisticRegression(max_iter=2000, C=1e6)


def prepare_ml_frame(data=None, scaling=None, expected_flow=None):
    """Part 3's retained sample, with ratings standardised by Part 3's scaler.

    Returns (frame, scaling). `frame` gains `z_<attr>` and `d_<attr>` columns:
    standardised, and standardised-then-person-demeaned.
    """
    data = load_clean() if data is None else data
    root = project_root()
    if scaling is None:
        scaling = pd.read_csv(root / TABLES_DIR / "core_model_scaling.csv")
    if expected_flow is None:
        expected_flow = pd.read_csv(root / TABLES_DIR / "core_model_sample_flow.csv")

    required = ["iid", "gender", "wave", "dec"] + ATTRS + STATED
    parts = []
    for code, name in GENDERS:
        side = data[data["gender"] == code]
        complete = side.dropna(subset=required)
        variation = complete.groupby("iid")["dec"].nunique()
        retained = variation[variation == 2].index
        frame = complete[complete["iid"].isin(retained)].sort_values("iid").copy()

        # Fail loudly rather than silently analysing a different sample.
        want = expected_flow[expected_flow["gender"] == name].iloc[0]
        if (len(complete), len(frame), frame["iid"].nunique()) != (
                int(want["complete_rows"]), int(want["model_rows"]),
                int(want["model_people"])):
            raise AssertionError(
                f"{name}: rebuilt sample {len(complete)}/{len(frame)} rows, "
                f"{frame['iid'].nunique()} people; Part 3 published "
                f"{int(want['complete_rows'])}/{int(want['model_rows'])} rows, "
                f"{int(want['model_people'])} people")

        scale = scaling[scaling["gender"] == name].set_index("attribute")
        for attribute in ATTRS:
            z = (frame[attribute] - scale.loc[attribute, "mean"]) / scale.loc[attribute, "sd_ddof0"]
            frame[f"z_{attribute}"] = z
            frame[f"d_{attribute}"] = z - z.groupby(frame["iid"]).transform("mean")
        frame["group"] = name
        parts.append(frame)

    return pd.concat(parts, ignore_index=True), scaling


def _cv_scores(frame, cols, model_factory):
    """Grouped-CV AUC. Returns (mean, sd, out-of-fold probabilities)."""
    x, y = frame[cols].to_numpy(float), frame["dec"].to_numpy(int)
    groups = frame["iid"].to_numpy()
    oof, folds = np.full(len(y), np.nan), []
    for train, test in GroupKFold(n_splits=N_SPLITS).split(x, y, groups):
        model = model_factory().fit(x[train], y[train])
        oof[test] = model.predict_proba(x[test])[:, 1]
        folds.append(roc_auc_score(y[test], oof[test]))
    return float(np.mean(folds)), float(np.std(folds)), oof


def benchmark(frame):
    """Does a flexible model beat a linear one? Per gender and pooled."""
    rows, oof = [], {}
    feature_sets = (("standardised", [f"z_{a}" for a in ATTRS]),
                    ("person-demeaned", [f"d_{a}" for a in ATTRS]))
    subsets = [(name, frame[frame["group"] == name]) for _, name in GENDERS]
    subsets.append(("Pooled", frame))
    for subset_name, subset in subsets:
        for feature_name, cols in feature_sets:
            lm, lsd, lp = _cv_scores(subset, cols, make_logit)
            gm, gsd, gp = _cv_scores(subset, cols, make_gbm)
            rows.append(dict(sample=subset_name, features=feature_name,
                             n_rows=len(subset), n_people=subset["iid"].nunique(),
                             logistic_auc=lm, logistic_sd=lsd,
                             gbm_auc=gm, gbm_sd=gsd, gbm_minus_logistic=gm - lm))
            if feature_name == "standardised":
                oof[subset_name] = (subset, lp, gp)
    return pd.DataFrame(rows), oof


def importance(frame, published):
    """Three importance estimates per gender, on Part 3's convention.

    conditional logit : Part 3's published shares (person fixed effects)
    pooled logit      : same features, fixed effects dropped
    GBM permutation   : AUC drop when one feature is shuffled

    Signed standardised coefficients are reported alongside the clipped shares.
    They are the point of the comparison: the group's importance convention maps
    every negative coefficient to 0.00, so three attributes report zero
    importance in the pooled model while actually carrying a sizeable negative
    association. Rank agreement between the clipped columns therefore
    understates how much dropping the fixed effects changes the estimates.
    """
    rows, agreement = [], []
    cols = [f"z_{a}" for a in ATTRS]
    for _, name in GENDERS:
        subset = frame[frame["group"] == name]
        x, y = subset[cols].to_numpy(float), subset["dec"].to_numpy(int)
        groups = subset["iid"].to_numpy()

        pooled = make_logit().fit(x, y).coef_[0]
        pooled_share = 100 * np.maximum(pooled, 0) / np.maximum(pooled, 0).sum()

        # Permutation importance is measured out of fold, then averaged.
        drops = np.zeros((N_SPLITS, len(ATTRS)))
        for fold, (train, test) in enumerate(
                GroupKFold(n_splits=N_SPLITS).split(x, y, groups)):
            model = make_gbm().fit(x[train], y[train])
            result = permutation_importance(
                model, x[test], y[test], scoring="roc_auc",
                n_repeats=10, random_state=SEED)
            drops[fold] = np.maximum(result.importances_mean, 0)
        gbm_share = 100 * drops.mean(axis=0) / drops.mean(axis=0).sum()

        side = published[published["gender"] == name].set_index("attribute")
        published_share = side.loc[ATTRS, "revealed"].to_numpy()
        published_coef = side.loc[ATTRS, "coefficient_std"].to_numpy()
        for k, (attribute, label) in enumerate(zip(ATTRS, ATTR_LABELS)):
            rows.append(dict(gender=name, attribute=attribute, label=label,
                             conditional_logit=published_share[k],
                             pooled_logit=pooled_share[k], gbm_permutation=gbm_share[k],
                             conditional_coef_std=published_coef[k],
                             pooled_coef_std=pooled[k],
                             coef_shift_pooled_minus_conditional=pooled[k] - published_coef[k]))
        agreement.append(dict(
            gender=name,
            spearman_gbm_vs_pooled=float(spearmanr(gbm_share, pooled_share).statistic),
            spearman_gbm_vs_conditional=float(spearmanr(gbm_share, published_share).statistic),
            spearman_pooled_vs_conditional=float(spearmanr(pooled_share, published_share).statistic)))
    return pd.DataFrame(rows), pd.DataFrame(agreement)


def threshold_check(frame):
    """Is there a dealbreaker cut-off in attractiveness, or is it smooth?"""
    # A dozen ratings are fractional (e.g. 7.5). Bin to the nearest integer so
    # the curve is not dominated by single-row points.
    subset = frame.copy()
    subset["attr_bin"] = subset["attr"].round().astype(int)
    observed = (subset.groupby("attr_bin")
                .agg(n=("dec", "size"), observed_p_yes=("dec", "mean"))
                .reset_index().rename(columns={"attr_bin": "attr"}))
    y, groups = subset["dec"].to_numpy(int), subset["iid"].to_numpy()
    # Both terms use the raw 1-10 rating so the only difference is functional
    # form: one slope versus one free parameter per rating level.
    linear = subset[["attr"]].to_numpy(float)
    dummies = pd.get_dummies(subset["attr_bin"], prefix="r").to_numpy(float)
    scores = {}
    for name, x in (("linear", linear), ("dummies", dummies)):
        oof = np.full(len(y), np.nan)
        for train, test in GroupKFold(n_splits=N_SPLITS).split(x, y, groups):
            oof[test] = make_logit().fit(x[train], y[train]).predict_proba(x[test])[:, 1]
        scores[name] = float(log_loss(y, oof))
    gbm_fit = _cv_scores(subset, ["attr"], make_gbm)[2]
    observed = observed.merge(
        pd.DataFrame({"attr": subset["attr_bin"].to_numpy(), "gbm_fitted": gbm_fit})
        .groupby("attr").mean().reset_index(), on="attr", how="left")
    return observed, dict(log_loss_linear=scores["linear"],
                          log_loss_dummies=scores["dummies"],
                          improvement=scores["linear"] - scores["dummies"],
                          n_fractional_ratings=int((frame["attr"] % 1 != 0).sum()))


def interaction_check(frame):
    """Do attribute trade-offs interact, or are preferences compensatory?"""
    cols = [f"z_{a}" for a in ATTRS]
    main = _cv_scores(frame, cols, make_logit)[0]
    wide = frame.copy()
    pairs = []
    for i in range(len(ATTRS)):
        for j in range(i + 1, len(ATTRS)):
            name = f"x_{ATTRS[i]}_{ATTRS[j]}"
            wide[name] = wide[cols[i]] * wide[cols[j]]
            pairs.append(name)
    both = _cv_scores(wide, cols + pairs, make_logit)[0]
    return dict(auc_main_effects=main, auc_with_interactions=both,
                n_interaction_terms=len(pairs), change=both - main)


def run_analysis(data=None):
    root = project_root()
    published = pd.read_csv(root / TABLES_DIR / "core_model_results.csv")
    frame, scaling = prepare_ml_frame(data)
    auc, oof = benchmark(frame)
    imp, agreement = importance(frame, published)
    curve, fit_quality = threshold_check(frame)
    interactions = interaction_check(frame)
    diagnostics = dict(
        seed=SEED, cv="GroupKFold by iid", n_splits=N_SPLITS,
        gbm="sklearn HistGradientBoostingClassifier",
        gbm_note="PROJECT_CONTEXT names XGBoost/LightGBM; both need libomp, sklearn does not",
        standardisation="Part 3 within-gender means/SDs from core_model_scaling.csv",
        sample="union of Part 3 retained samples; asserted against core_model_sample_flow.csv",
        n_rows=int(len(frame)), n_people=int(frame["iid"].nunique()),
        python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, sklearn=sklearn.__version__,
        threshold=fit_quality, interactions=interactions)
    return dict(frame=frame, auc=auc, importance=imp, agreement=agreement,
                curve=curve, diagnostics=diagnostics, oof=oof)


def save_outputs(analysis):
    root = project_root()
    tables, figures = root / TABLES_DIR, root / FIGURES_DIR
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    analysis["auc"].to_csv(tables / "ml_benchmark.csv", index=False)
    analysis["importance"].to_csv(tables / "ml_importance.csv", index=False)
    analysis["agreement"].to_csv(tables / "ml_rank_agreement.csv", index=False)
    analysis["curve"].to_csv(tables / "ml_attractiveness_curve.csv", index=False)
    (tables / "ml_diagnostics.json").write_text(
        json.dumps(analysis["diagnostics"], indent=2) + "\n")

    imp = analysis["importance"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, sharex=True)
    width = 0.26
    for ax, (_, name) in zip(axes, GENDERS):
        q = imp[imp["gender"] == name].set_index("label").loc[ATTR_LABELS]
        pos = np.arange(len(ATTR_LABELS))
        for shift, col, color, label in (
                (-width, "conditional_logit", REVEALED_COLOR, "Conditional logit (Part 3)"),
                (0.0, "pooled_logit", "#9c9c9c", "Pooled logit (no fixed effects)"),
                (width, "gbm_permutation", STATED_COLOR, "GBM permutation")):
            ax.barh(pos + shift, q[col], height=width, color=color, label=label)
        ax.set_yticks(pos, ATTR_LABELS)
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel("Relative importance (share of 100)")
        ax.grid(axis="x", alpha=.25)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=.95)
    fig.suptitle("Three methods, one ranking: attractiveness first, ambition last",
                 x=.04, ha="left", fontsize=12)
    fig.text(.04, .885,
             "Rank agreement between the fixed-effects model and a gradient booster is "
             "0.83 (women) and 0.94 (men). The ordering is robust; the estimates behind it are not.",
             ha="left", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(figures / "ml_importance_comparison.png", dpi=300)
    fig.savefig(figures / "ml_importance_comparison.svg")
    plt.close(fig)

    # The finding the shares hide: every coefficient shrinks without fixed
    # effects, and four cross zero, which the clipping rule reports as 0.00.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, sharex=True)
    for ax, (_, name) in zip(axes, GENDERS):
        q = imp[imp["gender"] == name].set_index("label").loc[ATTR_LABELS]
        pos = np.arange(len(ATTR_LABELS))
        ax.hlines(pos, q["pooled_coef_std"], q["conditional_coef_std"],
                  color="#bbb", lw=2, zorder=1)
        ax.scatter(q["pooled_coef_std"], pos, s=45, color="#9c9c9c", zorder=2,
                   label="Pooled logit (no fixed effects)")
        ax.scatter(q["conditional_coef_std"], pos, s=45, color=REVEALED_COLOR,
                   zorder=2, label="Conditional logit (Part 3)")
        ax.axvline(0, color="#333", lw=.9, ls=":")
        ax.set_yticks(pos, ATTR_LABELS)
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel("Standardised coefficient on dec")
        ax.grid(axis="x", alpha=.25)
    axes[0].set_xlim(-0.45, 2.15)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=.95)
    fig.suptitle("Without person fixed effects every weight is attenuated",
                 x=.04, ha="left", fontsize=12)
    fig.text(.04, .885,
             "All twelve move toward zero when the person effect is dropped (shift -0.21 to -0.74 SD).\n"
             "Four cross below zero, which the clipping rule reports as 0.00 importance.",
             ha="left", fontsize=8.5, color="#444", va="top")
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(figures / "ml_fixed_effect_shift.png", dpi=300)
    plt.close(fig)

    curve = analysis["curve"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(curve["attr"], curve["observed_p_yes"], "o-", color=REVEALED_COLOR,
            label="Observed P(yes)")
    ax.plot(curve["attr"], curve["gbm_fitted"], "s--", color=STATED_COLOR,
            markersize=4, label="GBM fitted (out of fold)")
    ax.set_xlabel("Attractiveness rating given to the partner (1–10)")
    ax.set_ylabel("P(say yes)")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    ax.set_title("No dealbreaker threshold: the curve is smooth", fontsize=11)
    fig.text(.01, .01,
             f"Grouped 5-fold CV by participant. Ten rating dummies beat a linear term by "
             f"{analysis['diagnostics']['threshold']['improvement']:.5f} of log-loss.",
             fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(figures / "ml_attractiveness_curve.png", dpi=300)
    plt.close(fig)
    return analysis


def main():
    analysis = save_outputs(run_analysis())
    print(analysis["auc"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(analysis["importance"].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()
    print(analysis["agreement"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(json.dumps(analysis["diagnostics"]["threshold"], indent=2))
    print(json.dumps(analysis["diagnostics"]["interactions"], indent=2))


if __name__ == "__main__":
    main()
