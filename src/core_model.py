"""Part 3: person fixed-effects conditional logit and stated/revealed gaps.

Run from the repository root: python -m src.core_model
The notebook calls the same functions. Only the six partner ratings predict dec.
Standardisation is within each gender's retained model sample (ddof=0) and is
recomputed within each participant-bootstrap sample. Importance follows the
project convention: 100 * max(standardised coefficient, 0) / sum(positive coefs).
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import statsmodels
from statsmodels.discrete.conditional_models import ConditionalLogit
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import (ATTRS, ATTR_LABELS, STATED, SEED, STATED_COLOR,
                        REVEALED_COLOR, CLEAN_PATH, FIGURES_DIR, TABLES_DIR)
from .io import load_clean, project_root


@dataclass
class Sample:
    frame: pd.DataFrame
    x_centered: np.ndarray
    y: np.ndarray
    offsets: np.ndarray
    yes_counts: np.ndarray
    stated: np.ndarray
    row_counts: np.ndarray
    sums: np.ndarray
    sums_squared: np.ndarray


def prepare_sample(data):
    """Retain complete cases and people with both outcomes; count exclusions."""
    required = ["iid", "gender", "wave", "dec"] + ATTRS + STATED
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not data["dec"].dropna().isin([0, 1]).all():
        raise ValueError("dec must be binary")
    if not data[ATTRS].stack().dropna().between(1, 10).all():
        raise ValueError("Read the unscaled cleaned dataset, not z-scored data")
    if (data.groupby("iid")[STATED + ["gender", "wave"]].nunique() > 1).any().any():
        raise ValueError("Person-level fields vary within iid")
    complete = data.dropna(subset=required).copy()
    if (complete[STATED] < 0).any().any() or (complete[STATED].sum(axis=1) <= 0).any():
        raise ValueError("Stated weights must be nonnegative with positive totals")
    variation = complete.groupby("iid")["dec"].nunique()
    retained = variation[variation == 2].index
    frame = complete[complete["iid"].isin(retained)].sort_values("iid").copy()
    if frame.empty:
        raise ValueError("No people with both yes and no decisions")
    frame[STATED] = frame[STATED].div(frame[STATED].sum(axis=1), axis=0) * 100
    grouped = frame.groupby("iid", sort=True)
    counts = grouped.size().to_numpy(dtype=np.int64)
    raw_x = frame[ATTRS].to_numpy(dtype=float, copy=True)
    # Person centering leaves the conditional likelihood unchanged and keeps
    # the exact recursion numerically well-behaved. SDs use uncentered ratings.
    centered = raw_x - grouped[ATTRS].transform("mean").to_numpy()
    sample = Sample(
        frame=frame, x_centered=np.ascontiguousarray(centered),
        y=frame["dec"].to_numpy(dtype=float),
        offsets=np.r_[0, np.cumsum(counts)].astype(np.int64),
        yes_counts=grouped["dec"].sum().to_numpy(dtype=np.int64),
        stated=frame.drop_duplicates("iid")[STATED].to_numpy(dtype=float),
        row_counts=counts, sums=grouped[ATTRS].sum().to_numpy(),
        sums_squared=frame[ATTRS].pow(2).groupby(frame["iid"], sort=True).sum().to_numpy(),
    )
    flow = dict(raw_rows=len(data), raw_people=data["iid"].nunique(),
                complete_rows=len(complete), complete_people=complete["iid"].nunique(),
                excluded_missing_rows=len(data)-len(complete),
                excluded_constant_choice_people=len(variation)-len(retained),
                excluded_constant_choice_rows=len(complete)-len(frame),
                model_rows=len(frame), model_people=len(retained),
                model_yes_rate=float(frame["dec"].mean()))
    return sample, flow


def rating_scaler(sample, multiplicity):
    """Means and population SDs of rows in a resampled collection of people."""
    count = multiplicity @ sample.row_counts
    mean = multiplicity @ sample.sums / count
    variance = multiplicity @ sample.sums_squared / count - mean**2
    if np.any(variance <= 1e-12):
        raise ValueError("A rating has zero variance")
    return mean, np.sqrt(variance)


@njit
def conditional_objective(beta, x, y, offsets, yes_counts, multiplicity):
    """Exact negative conditional log-likelihood and analytic gradient.

    For each person, condition on m yes decisions among n dates. D[k] sums
    exp(total utility) over all subsets of k dates; G[k] is its derivative.
    A descending recursion computes the denominator without enumerating all
    subsets. Group multiplicities implement a whole-person cluster bootstrap,
    equivalent to duplicating sampled people with fresh stratum IDs.
    """
    p = x.shape[1]
    loss = 0.0
    gradient = np.zeros(p)
    for i in range(len(yes_counts)):
        if multiplicity[i] == 0:
            continue
        m = yes_counts[i]
        denominator = np.zeros(m+1)
        denominator[0] = 1.0
        derivative = np.zeros((m+1, p))
        observed_utility = 0.0
        observed_gradient = np.zeros(p)
        for j in range(offsets[i], offsets[i+1]):
            eta = np.dot(x[j], beta)
            weight = np.exp(eta)
            observed_utility += y[j]*eta
            observed_gradient += y[j]*x[j]
            for k in range(min(m, j-offsets[i]+1), 0, -1):
                derivative[k] += weight*(derivative[k-1] + denominator[k-1]*x[j])
                denominator[k] += weight*denominator[k-1]
        loss += multiplicity[i]*(np.log(denominator[m])-observed_utility)
        gradient += multiplicity[i]*(derivative[m]/denominator[m]-observed_gradient)
    return loss, gradient


def fit_sample(sample, multiplicity, start_raw=None):
    """Fit standardised predictors, retaining signed coefficients for audit."""
    mean, sd = rating_scaler(sample, multiplicity)
    x = np.ascontiguousarray(sample.x_centered / sd)
    start = np.zeros(len(ATTRS)) if start_raw is None else start_raw * sd
    args = (x, sample.y, sample.offsets, sample.yes_counts, multiplicity)
    fitted = minimize(conditional_objective, start, args=args, jac=True,
                      method="BFGS", options={"gtol": 1e-6, "maxiter": 300})
    gradient = float(np.max(np.abs(fitted.jac)))
    if not np.isfinite(fitted.fun) or gradient > 1e-4:
        raise RuntimeError(f"Fit did not converge: {fitted.message}; gradient={gradient}")
    return dict(beta=fitted.x, beta_raw=fitted.x/sd, mean=mean, sd=sd,
                max_abs_gradient=gradient, loglike=-float(fitted.fun))


def revealed_weights(beta):
    """Project convention; clipping affects importance only, not fitted betas."""
    positive = np.maximum(beta, 0)
    if not np.isfinite(positive).all() or positive.sum() <= 1e-12:
        raise ValueError("No positive coefficients to normalise")
    return 100 * positive / positive.sum()


def run_analysis(data=None, n_bootstrap=1000):
    """Return results, sample flow, scaling, and convergence diagnostics.

    Each draw recomputes standardisation, refits, clips/normalises coefficients,
    and recomputes the stated mean using the same sampled participants.
    Intervals are pointwise percentile 95% intervals, not familywise tests.
    """
    if n_bootstrap < 2:
        raise ValueError("Use at least two bootstrap draws; final analysis uses 1000")
    data = load_clean() if data is None else data
    rng = np.random.default_rng(SEED)
    results, flows, scalers = [], [], []
    diagnostics = {"seed": SEED, "requested_draws_per_gender": n_bootstrap,
                   "standardisation": "within gender, retained model rows, ddof=0; refit each draw",
                   "bootstrap_unit": "iid", "interval": "pointwise percentile 95%",
                   "python": platform.python_version(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scipy": scipy.__version__,
                   "statsmodels": statsmodels.__version__, "groups": {}}
    if not data["gender"].dropna().isin([0, 1]).all():
        raise ValueError("Expected gender codes 0 and 1")
    for code, name in [(0, "Women"), (1, "Men")]:
        sample, flow = prepare_sample(data[data["gender"] == code])
        flows.append({"gender": name, **flow})
        n_people = len(sample.stated)
        base = fit_sample(sample, np.ones(n_people))
        standardized_x = (sample.frame[ATTRS].to_numpy()-base["mean"]) / base["sd"]
        reference_model = ConditionalLogit(sample.y, standardized_x, groups=sample.frame["iid"])
        reference = reference_model.fit(method="bfgs", disp=False, maxiter=300)
        difference = float(np.max(np.abs(base["beta"] - reference.params)))
        if difference > 0.001:
            raise AssertionError(f"Statsmodels validation differs by {difference}")
        reference_ll = reference_model.loglike(base["beta"])
        if abs(reference_ll - base["loglike"]) > 1e-7:
            raise AssertionError("Exact likelihood differs from statsmodels")
        stated = sample.stated.mean(axis=0)
        revealed = revealed_weights(base["beta"])
        draws = []
        failures = []
        for b in range(n_bootstrap):
            mult = rng.multinomial(n_people, np.full(n_people, 1/n_people)).astype(float)
            try:
                fitted = fit_sample(sample, mult, base["beta_raw"])
                draw_stated = np.average(sample.stated, axis=0, weights=mult)
                draw_revealed = revealed_weights(fitted["beta"])
                draws.append(np.stack([draw_stated, draw_revealed,
                                       draw_revealed-draw_stated, fitted["beta"]]))
            except (RuntimeError, ValueError) as error:
                failures.append({"draw": b, "error": str(error)})
            if (b+1) % 250 == 0:
                print(f"{name}: {b+1}/{n_bootstrap} participant resamples; failures={len(failures)}", flush=True)
        if len(failures) > 0.05*n_bootstrap:
            raise RuntimeError(f"Too many failed bootstrap fits: {len(failures)}")
        if failures:
            warnings.warn(f"{name}: {len(failures)} failed fits; see diagnostics", stacklevel=2)
        draw_array = np.asarray(draws)
        ci = np.quantile(draw_array, [0.025, 0.975], axis=0)
        for k, (attribute, label) in enumerate(zip(ATTRS, ATTR_LABELS)):
            row = dict(gender=name, attribute=attribute, label=label,
                       coefficient_std=base["beta"][k], coefficient_raw=base["beta_raw"][k],
                       odds_ratio_per_sd=np.exp(base["beta"][k]),
                       stated=stated[k], revealed=revealed[k], gap_pp=revealed[k]-stated[k])
            for j, metric in enumerate(["stated", "revealed", "gap", "coefficient_std"]):
                row[f"{metric}_ci_low"] = ci[0, j, k]
                row[f"{metric}_ci_high"] = ci[1, j, k]
            row["gap_ci_excludes_zero"] = bool(ci[0, 2, k] > 0 or ci[1, 2, k] < 0)
            results.append(row)
            scalers.append(dict(gender=name, attribute=attribute, mean=base["mean"][k],
                                sd_ddof0=base["sd"][k]))
        diagnostics["groups"][name] = dict(successful_draws=len(draws), failed_draws=failures,
            statsmodels_max_coefficient_difference=difference,
            statsmodels_loglike_difference=float(abs(reference_ll-base["loglike"])),
            max_abs_gradient=base["max_abs_gradient"])
    return dict(results=pd.DataFrame(results), flow=pd.DataFrame(flows),
                scaling=pd.DataFrame(scalers), diagnostics=diagnostics)


def save_outputs(analysis):
    """Write only Part 3 outputs, leaving cleaning and others' outputs intact."""
    root = project_root()
    figures, tables = root / FIGURES_DIR, root / TABLES_DIR
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for key, suffix in [("results", "results"), ("flow", "sample_flow"), ("scaling", "scaling")]:
        analysis[key].to_csv(tables / f"core_model_{suffix}.csv", index=False)
    diagnostics = dict(analysis["diagnostics"])
    diagnostics["input_sha256"] = hashlib.sha256((root / CLEAN_PATH).read_bytes()).hexdigest()
    (tables / "core_model_diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    results, flow = analysis["results"], analysis["flow"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 7.5), sharey=True, sharex=True)
    y = np.arange(len(ATTRS))
    max_upper = max(results["revealed_ci_high"].max(), results["stated_ci_high"].max())
    for ax, name in zip(axes, ["Women", "Men"]):
        q = results[results["gender"] == name]
        for shift, metric, color, label in [(-.17, "stated", STATED_COLOR, "Stated"),
                                            (.17, "revealed", REVEALED_COLOR, "Revealed")]:
            ax.barh(y+shift, q[metric], height=.30, color=color, label=label)
            # Draw interval endpoints directly: a percentile CI need not bracket
            # the original point estimate, especially after clipping at zero.
            ax.hlines(y+shift, q[f"{metric}_ci_low"], q[f"{metric}_ci_high"], color="#263747", lw=1)
            ax.vlines(q[f"{metric}_ci_low"], y+shift-.05, y+shift+.05, color="#263747", lw=1)
            ax.vlines(q[f"{metric}_ci_high"], y+shift-.05, y+shift+.05, color="#263747", lw=1)
        ax.set_yticks(y, ATTR_LABELS)
        ax.set_ylim(5.65, -.65)
        ax.set_xlim(0, np.ceil((max_upper+2)/10)*10)
        ax.set_xlabel("Relative importance (out of 100)")
        people = int(flow.loc[flow.gender == name, "model_people"].iloc[0])
        ax.set_title(f"{name} · {people} participants", loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Attractiveness carries more weight in choices than in stated preferences",
                 x=.04, ha="left", fontsize=17, fontweight="bold")
    fig.text(.04, .91, f'Conditional logit by gender · {int(flow.model_rows.sum()):,} decisions · person fixed effects',
             fontsize=12, color="#526273")
    fig.text(.04, .035,
             f'95% participant-bootstrap intervals; {diagnostics["requested_draws_per_gender"]:,} resamples per gender; seed {SEED}.\n'
             'Revealed = positive standardized coefficients normalized to 100; negative coefficients clipped to 0.\n'
             'Descriptive, model-dependent importance; not causal effects. Source: Columbia Speed Dating data.',
             fontsize=9.5, color="#526273")
    fig.subplots_adjust(left=.15, right=.98, top=.82, bottom=.20, wspace=.12)
    fig.savefig(figures / "core_model_stated_vs_revealed.png", dpi=300)
    svg_path = figures / "core_model_stated_vs_revealed.svg"
    fig.savefig(svg_path)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines())+"\n")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), sharex=True, sharey=True)
    for ax, name in zip(axes, ["Women", "Men"]):
        q = results[results.gender == name]
        ax.hlines(y, q.gap_ci_low, q.gap_ci_high, color=REVEALED_COLOR, lw=2)
        ax.plot(q.gap_pp, y, "o", color=REVEALED_COLOR)
        ax.axvline(0, ls="--", color="#526273", lw=1)
        ax.set_yticks(y, ATTR_LABELS)
        ax.set_ylim(5.65, -.65)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Revealed minus stated (percentage points)")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("How large is the stated–revealed gap?", fontsize=18, fontweight="bold")
    fig.text(.04, .025, 'Pointwise 95% participant-bootstrap intervals. An interval crossing zero is inconclusive.\n'
             'Intervals are not adjusted for multiple comparisons; importance uses clipped standardized coefficients.', fontsize=9.5)
    fig.tight_layout(rect=[0,.08,1,.94])
    fig.savefig(figures / "core_model_gap_intervals.png", dpi=300)
    plt.close(fig)
    return diagnostics


def main():
    analysis = run_analysis()
    save_outputs(analysis)
    print(analysis["results"][["gender", "label", "stated", "revealed", "gap_pp",
                               "gap_ci_low", "gap_ci_high"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
