"""Shared helpers for the extended analyses (docs/extended_analysis_report.md).

Everything here builds on Part 3: the same retained sample, the same exact
conditional likelihood and the same importance rule. The new material is the
feature construction, which has one rule worth reading before changing it:

    A feature used to RANK a person's list must not contain that row's own
    outcome, even by subtraction. A leave-one-out mean does: within a list, it
    rises exactly when this row's outcome was low, so ranking on it selects
    people who rejected you. That error produced a "worse than random"
    reciprocal rule during development. Split-fold and history/future features
    below avoid it. Leave-one-out consensus is kept for regressions only.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

from .constants import ATTRS, SEED, TABLES_DIR
from .core_model import conditional_objective, revealed_weights
from .customer_value import build_model_sample
from .io import load_clean, project_root

GENDERS = ((0, "Women"), (1, "Men"))
N_BOOT = 1000
CONSENSUS = [f"c_{a}" for a in ATTRS]
SPLIT_FOLD = [f"sf_{a}" for a in ATTRS]
HISTORY = [f"h_{a}" for a in ATTRS]
PAIR = ["int_corr", "agegap", "samerace"]
PARTNER_STATED = {"attr": "pf_o_att", "sinc": "pf_o_sin", "intel": "pf_o_int",
                  "fun": "pf_o_fun", "amb": "pf_o_amb", "shar": "pf_o_sha"}


def core_sample(data=None):
    """Part 3's retained rows, asserted against its published sample flow."""
    frame = build_model_sample(load_clean() if data is None else data)
    flow = pd.read_csv(project_root() / TABLES_DIR / "core_model_sample_flow.csv")
    expected = (int(flow["model_rows"].sum()), int(flow["model_people"].sum()))
    found = (len(frame), frame["iid"].nunique())
    if found != expected:
        raise ValueError(f"Sample {found} differs from Part 3 {expected}")
    return frame


def normalised_weights(people, suffix):
    """Six preference columns for a question suffix, rescaled to sum to 100."""
    columns = [f"{a}{suffix}" for a in ATTRS]
    values = people[columns].astype(float)
    totals = values.sum(axis=1, min_count=len(ATTRS))
    out = values.div(totals.where(totals > 0), axis=0) * 100
    return out.set_axis(ATTRS, axis=1)


# ---------------------------------------------------------------- likelihood


@dataclass
class Strata:
    """One conditional-logit dataset: rows grouped by person, sorted by iid."""
    raw: np.ndarray
    centred: np.ndarray
    y: np.ndarray
    offsets: np.ndarray
    yes_counts: np.ndarray
    row_counts: np.ndarray
    ids: np.ndarray
    columns: list


@dataclass
class Fit:
    beta: np.ndarray
    beta_raw: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    loglike: float

    def score(self, frame, columns):
        """Linear utility for new rows, on this fit's own standardisation."""
        return (frame[columns].to_numpy(dtype=float) - self.mean) / self.sd @ self.beta


def make_strata(frame, columns):
    """Drop incomplete rows, then people with no yes/no variation left."""
    columns = list(columns)
    rows = frame.dropna(subset=columns + ["dec"])
    varied = rows.groupby("iid")["dec"].transform("nunique") == 2
    rows = rows[varied].sort_values("iid", kind="stable")
    grouped = rows.groupby("iid", sort=True)
    counts = grouped.size().to_numpy(dtype=np.int64)
    raw = rows[columns].to_numpy(dtype=float)
    centred = raw - grouped[columns].transform("mean").to_numpy()
    return Strata(raw=raw, centred=np.ascontiguousarray(centred),
                  y=rows["dec"].to_numpy(dtype=float),
                  offsets=np.r_[0, np.cumsum(counts)].astype(np.int64),
                  yes_counts=grouped["dec"].sum().to_numpy(dtype=np.int64),
                  row_counts=counts, ids=grouped["dec"].sum().index.to_numpy(),
                  columns=columns)


def fit_clogit(strata, multiplicity=None, start=None):
    """Exact conditional logit on Part 3's likelihood.

    `multiplicity` weights people, which implements both the whole-person
    bootstrap and inverse-probability weighting. Scaling follows the weights,
    as in Part 3's bootstrap.
    """
    n_people = len(strata.yes_counts)
    weights = np.ones(n_people) if multiplicity is None else np.asarray(multiplicity, float)
    rows = np.repeat(weights, strata.row_counts)
    mean = rows @ strata.raw / rows.sum()
    sd = np.sqrt(np.maximum(rows @ strata.raw**2 / rows.sum() - mean**2, 1e-12))
    x = np.ascontiguousarray(strata.centred / sd)
    begin = np.zeros(x.shape[1]) if start is None else start
    fitted = minimize(conditional_objective, begin, jac=True, method="BFGS",
                      args=(x, strata.y, strata.offsets, strata.yes_counts, weights),
                      options={"gtol": 1e-5, "maxiter": 300})
    return Fit(fitted.x, fitted.x / sd, mean, sd, -float(fitted.fun))


def bootstrap_fits(strata, n_boot, seed=SEED):
    """Whole-person bootstrap. Returns standardised and raw-scale coefficients."""
    rng = np.random.default_rng(seed)
    base = fit_clogit(strata)
    n_people = len(strata.yes_counts)
    std, raw = np.zeros((n_boot, len(base.beta))), np.zeros((n_boot, len(base.beta)))
    for b in range(n_boot):
        multiplicity = np.bincount(rng.integers(0, n_people, n_people),
                                   minlength=n_people).astype(float)
        fit = fit_clogit(strata, multiplicity, base.beta)
        std[b], raw[b] = fit.beta, fit.beta_raw
    return base, std, raw


def importance(beta):
    """Project rule: clip negatives to zero, rescale to 100."""
    return revealed_weights(np.asarray(beta, dtype=float))


def importance_draws(draws):
    positive = np.maximum(draws, 0)
    return 100 * positive / positive.sum(axis=1, keepdims=True)


@njit
def person_loglik(beta, x, y, offsets, yes_counts):
    """Each person's exact conditional log-likelihood (latent-class E step)."""
    out = np.zeros(len(yes_counts))
    for i in range(len(yes_counts)):
        m = yes_counts[i]
        denominator = np.zeros(m + 1)
        denominator[0] = 1.0
        observed = 0.0
        for j in range(offsets[i], offsets[i + 1]):
            eta = np.dot(x[j], beta)
            weight = np.exp(eta)
            observed += y[j] * eta
            for k in range(min(m, j - offsets[i] + 1), 0, -1):
                denominator[k] += weight * denominator[k - 1]
        out[i] = observed - np.log(denominator[m])
    return out


# ------------------------------------------------------------------ features


def add_consensus(data):
    """Mean rating a partner got from everyone except this rater.

    Regressions only: see the module docstring before ranking on these.
    """
    out = data.copy()
    for attribute in ATTRS:
        grouped = out.groupby("pid")[attribute]
        total, count = grouped.transform("sum"), grouped.transform("count")
        own, present = out[attribute].fillna(0), out[attribute].notna().astype(int)
        out[f"c_{attribute}"] = (total - own) / (count - present).replace(0, np.nan)
    return out


def add_pair_features(frame):
    """Pair facts a platform knows before a meeting, median-filled."""
    out = frame.copy()
    out["agegap"] = (out["age"] - out["age_o"]).abs()
    for column in PAIR:
        out[column] = out[column].fillna(out[column].median())
    return out


def add_split_fold(data, seed=SEED):
    """Features for a target, built only from raters in the opposite fold.

    Each person gets a random fold. For rater i, target j's features come from
    j's raters in the other fold, so no row's own label enters its features and
    the fold is constant across i's list. `j_sel` is j's own yes rate, measured
    only on j's partners in the fold opposite to i.
    """
    out = data.copy()
    rng = np.random.default_rng(seed)
    people = out["iid"].unique()
    fold = pd.Series(rng.integers(0, 2, len(people)), index=people)
    out["fold"] = out["iid"].map(fold)
    for column in [*ATTRS, "dec"]:
        means = out.groupby(["pid", "fold"])[column].mean().rename(f"sf_{column}").reset_index()
        means["fold"] = 1 - means["fold"]
        out = out.merge(means, on=["pid", "fold"], how="left")
    sides = out[["iid", "pid", "dec"]].assign(partner_fold=out["pid"].map(fold))
    selectivity = (sides.groupby(["iid", "partner_fold"])["dec"].mean()
                   .rename("j_sel").reset_index())
    selectivity["fold"] = 1 - selectivity["partner_fold"]
    selectivity = selectivity.rename(columns={"iid": "pid"})[["pid", "fold", "j_sel"]]
    out = out.merge(selectivity, on=["pid", "fold"], how="left")
    out["j_sel"] = out["j_sel"].fillna(out["dec"].mean())
    return out


def split_history(data):
    """Early speed dates become history; later pairs are candidates.

    Every feature comes from history, so it is constant per person and holds
    no candidate's outcome. This is the design for anything that compares
    options across partners (market simulator), where a fold-dependent
    selectivity feature would steer choices towards a person's low-yes fold.
    Returns candidate rows that have complete history features.
    """
    frame = data.dropna(subset=["order", "dec", "pid"]).copy()
    cut = frame.groupby("wave")["order"].transform(lambda s: s.max() / 2)
    early = frame[frame["order"] <= cut]
    chooser = early.groupby("iid")["dec"].mean().rename("h_sel")
    received = {"h_rcv": ("dec", "mean"), **{f"h_{a}": (a, "mean") for a in ATTRS}}
    target = early.groupby("pid").agg(**received)
    later = (frame[frame["order"] > cut]
             .merge(chooser, left_on="iid", right_index=True, how="inner")
             .merge(target, left_on="pid", right_index=True, how="inner"))
    return later.dropna(subset=[*HISTORY, "h_rcv", "h_sel"])


# ----------------------------------------------------------------- inference


def p_from_draws(draws):
    """Two-sided bootstrap p: how readily the interval reaches zero."""
    draws = np.asarray(draws, dtype=float)
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())
                     + 1 / len(draws)))


def summarise_draws(draws):
    return dict(ci_low=float(np.quantile(draws, 0.025)),
                ci_high=float(np.quantile(draws, 0.975)),
                p=p_from_draws(draws))


def boot_mean(values, n_boot=N_BOOT, seed=SEED):
    """Bootstrap one value per person (or per event)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    return dict(estimate=float(values.mean()), **summarise_draws(values[idx].mean(axis=1)))


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(adjusted, 1.0)
    return out


def demean(frame, columns):
    """Within-person values: subtracting a person's mean absorbs their intercept."""
    return frame[columns] - frame.groupby("iid")[columns].transform("mean")


def relative_weights(x, y):
    """Johnson relative weights: importance summing to R², with no clipping."""
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    target = (y - y.mean()) / y.std()
    corr = np.corrcoef(z.T)
    eigenvalues, vectors = np.linalg.eigh(corr)
    root = vectors @ np.diag(np.sqrt(eigenvalues)) @ vectors.T
    betas = np.linalg.solve(root, z.T @ target / len(target))
    return root**2 @ betas**2


