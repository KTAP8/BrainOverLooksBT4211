"""Part 6: customer value of ranking by revealed vs stated preferences."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import (
    ATTRS,
    FIGURES_DIR,
    REVEALED_COLOR,
    SEED,
    STATED,
    STATED_COLOR,
    TABLES_DIR,
)
from .io import load_clean, project_root


def retain_core_model_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Same row filter as Part 3 prepare_sample (without logit-only arrays)."""
    required = ["iid", "gender", "wave", "dec"] + ATTRS + STATED
    complete = data.dropna(subset=required).copy()
    variation = complete.groupby("iid")["dec"].nunique()
    retained = variation[variation == 2].index
    frame = complete[complete["iid"].isin(retained)].sort_values("iid").copy()
    if frame.empty:
        raise ValueError("No people with both yes and no decisions")
    frame[STATED] = frame[STATED].div(frame[STATED].sum(axis=1), axis=0) * 100
    return frame


def load_revealed_weights(results_path: Path | None = None) -> dict[int, np.ndarray]:
    """Gender code -> length-6 revealed weight vector (sums to 100)."""
    root = project_root()
    path = results_path or root / "outputs/tables/core_model_results.csv"
    results = pd.read_csv(path)
    weights: dict[int, np.ndarray] = {}
    for code, name in [(0, "Women"), (1, "Men")]:
        sub = results[results["gender"] == name].set_index("attribute")
        if len(sub) != len(ATTRS):
            raise ValueError(f"Expected {len(ATTRS)} attributes for {name}")
        weights[code] = sub.loc[ATTRS, "revealed"].to_numpy(dtype=float)
    return weights


def build_model_sample(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same retained rows as Part 3 conditional logit (both genders)."""
    data = load_clean() if data is None else data
    parts = []
    for code in (0, 1):
        parts.append(retain_core_model_sample(data[data["gender"] == code]))
    return pd.concat(parts, ignore_index=True)


def add_match_scores(frame: pd.DataFrame, revealed: dict[int, np.ndarray]) -> pd.DataFrame:
    """Linear match indices using stated *1_1 and gender-specific revealed weights."""
    out = frame.copy()
    stated_w = out[STATED].to_numpy(dtype=float)
    out["score_stated"] = (stated_w * out[ATTRS].to_numpy(dtype=float)).sum(axis=1)
    rev = np.vstack([revealed[int(g)] for g in out["gender"].to_numpy()])
    out["score_revealed"] = (rev * out[ATTRS].to_numpy(dtype=float)).sum(axis=1)
    return out


def topk_yes_rates(
    frame: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 3),
) -> pd.DataFrame:
    """Per-person then pooled yes rates for top-k under each scoring rule."""
    rows = []
    for iid, sub in frame.groupby("iid", sort=True):
        gender = int(sub["gender"].iloc[0])
        n = len(sub)
        for k in k_values:
            kk = min(k, n)
            for rule, col in [("stated", "score_stated"), ("revealed", "score_revealed")]:
                top = sub.nlargest(kk, col)
                rows.append(
                    dict(
                        iid=iid,
                        gender=gender,
                        k=k,
                        k_effective=kk,
                        rule=rule,
                        yes_rate=float(top["dec"].mean()),
                        n_dates=n,
                    )
                )
    person = pd.DataFrame(rows)
    summary = (
        person.groupby(["gender", "k", "rule"], as_index=False)
        .agg(
            mean_yes_rate=("yes_rate", "mean"),
            n_people=("iid", "nunique"),
        )
    )
    summary["gender_label"] = summary["gender"].map({0: "Women", 1: "Men"})
    overall = (
        person.groupby(["k", "rule"], as_index=False)
        .agg(mean_yes_rate=("yes_rate", "mean"), n_people=("iid", "nunique"))
    )
    overall["gender"] = -1
    overall["gender_label"] = "All"
    summary = pd.concat([summary, overall], ignore_index=True)
    pivot = summary.pivot_table(
        index=["gender_label", "k", "n_people"],
        columns="rule",
        values="mean_yes_rate",
    ).reset_index()
    pivot["lift_pp"] = pivot["revealed"] - pivot["stated"]
    pivot["lift_ratio"] = pivot["revealed"] / pivot["stated"]
    return person, pivot


def top1_detail(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-person stated vs revealed #1 partner (one row per iid)."""
    rows = []
    for iid, sub in scored.groupby("iid", sort=True):
        stated_row = sub.nlargest(1, "score_stated").iloc[0]
        revealed_row = sub.nlargest(1, "score_revealed").iloc[0]
        same = int(stated_row["pid"] == revealed_row["pid"])
        rows.append(
            dict(
                iid=iid,
                gender=int(stated_row["gender"]),
                same_top1_partner=same,
                top1_changed=int(not same),
                stated_top1_yes=int(stated_row["dec"]),
                revealed_top1_yes=int(revealed_row["dec"]),
            )
        )
    return pd.DataFrame(rows)


def top1_agreement_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate top-1 agreement for FAQ / CSV export."""
    n = len(detail)
    same = detail["same_top1_partner"].astype(bool)
    diff = ~same
    return pd.DataFrame(
        [
            dict(
                segment="all",
                n_people=n,
                share_same_top1=detail["same_top1_partner"].mean(),
                stated_top1_yes_rate=detail["stated_top1_yes"].mean(),
                revealed_top1_yes_rate=detail["revealed_top1_yes"].mean(),
            ),
            dict(
                segment="same_top1",
                n_people=int(same.sum()),
                share_same_top1=1.0,
                stated_top1_yes_rate=detail.loc[same, "stated_top1_yes"].mean(),
                revealed_top1_yes_rate=detail.loc[same, "revealed_top1_yes"].mean(),
            ),
            dict(
                segment="different_top1",
                n_people=int(diff.sum()),
                share_same_top1=0.0,
                stated_top1_yes_rate=detail.loc[diff, "stated_top1_yes"].mean(),
                revealed_top1_yes_rate=detail.loc[diff, "revealed_top1_yes"].mean(),
            ),
        ]
    )


def top1_agreement(scored: pd.DataFrame) -> pd.DataFrame:
    """Compare stated vs revealed #1 partner per person (FAQ for slide 8)."""
    return top1_agreement_summary(top1_detail(scored))


def _segment_pivot(person: pd.DataFrame, segment_label: str) -> pd.DataFrame:
    summary = (
        person.groupby(["gender", "k", "rule"], as_index=False)
        .agg(mean_yes_rate=("yes_rate", "mean"), n_people=("iid", "nunique"))
    )
    summary["gender_label"] = summary["gender"].map({0: "Women", 1: "Men"})
    overall = (
        person.groupby(["k", "rule"], as_index=False)
        .agg(mean_yes_rate=("yes_rate", "mean"), n_people=("iid", "nunique"))
    )
    overall["gender"] = -1
    overall["gender_label"] = "All"
    summary = pd.concat([summary, overall], ignore_index=True)
    pivot = summary.pivot_table(
        index=["gender_label", "k", "n_people"],
        columns="rule",
        values="mean_yes_rate",
    ).reset_index()
    pivot["lift_pp"] = pivot["revealed"] - pivot["stated"]
    pivot["lift_ratio"] = pivot["revealed"] / pivot["stated"]
    pivot["segment"] = segment_label
    return pivot


def summary_by_top1_segment(
    person: pd.DataFrame,
    top1: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 3),
) -> pd.DataFrame:
    """Top-k lift summaries for all, same #1, and different #1 segments."""
    merged = person.merge(top1[["iid", "same_top1_partner", "top1_changed"]], on="iid")
    parts = [_segment_pivot(merged, "all")]
    parts.append(_segment_pivot(merged[merged["same_top1_partner"] == 1], "same_top1"))
    parts.append(_segment_pivot(merged[merged["top1_changed"] == 1], "different_top1"))
    out = pd.concat(parts, ignore_index=True)
    return out[out["k"].isin(k_values)].copy()


def bootstrap_lift(
    person: pd.DataFrame,
    k: int,
    n_draws: int = 1000,
    seed: int = SEED,
    iids: pd.Index | np.ndarray | None = None,
    segment: str = "all",
) -> dict:
    """Cluster bootstrap on iid for top-k lift (revealed minus stated yes rate)."""
    rng = np.random.default_rng(seed)
    sub = person[person["k"] == k]
    if iids is not None:
        sub = sub[sub["iid"].isin(iids)]
    wide = sub.pivot(index="iid", columns="rule", values="yes_rate")[["stated", "revealed"]]
    n = len(wide)
    if n == 0:
        raise ValueError(f"No people for bootstrap segment={segment!r}, k={k}")
    idx = rng.integers(0, n, size=(n_draws, n))
    stated = wide["stated"].to_numpy()
    revealed = wide["revealed"].to_numpy()
    lifts = revealed[idx].mean(axis=1) - stated[idx].mean(axis=1)
    arr = lifts
    return {
        "segment": segment,
        "k": k,
        "mean_lift_pp": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_people": n,
        "n_draws": n_draws,
    }


def run_customer_value(
    data: pd.DataFrame | None = None,
    n_bootstrap: int = 1000,
) -> dict:
    """Run simulation aligned with Part 3 sample."""
    frame = build_model_sample(data)
    revealed = load_revealed_weights()
    scored = add_match_scores(frame, revealed)
    person, summary = topk_yes_rates(scored)
    detail = top1_detail(scored)
    agreement = top1_agreement_summary(detail)
    summary_segment = summary_by_top1_segment(person, detail)
    changed_iids = detail.loc[detail["top1_changed"] == 1, "iid"]
    boot = [
        bootstrap_lift(person, k=1, n_draws=n_bootstrap, segment="all"),
        bootstrap_lift(person, k=3, n_draws=n_bootstrap, segment="all"),
        bootstrap_lift(
            person,
            k=1,
            n_draws=n_bootstrap,
            iids=changed_iids,
            segment="different_top1",
            seed=SEED + 1,
        ),
        bootstrap_lift(
            person,
            k=3,
            n_draws=n_bootstrap,
            iids=changed_iids,
            segment="different_top1",
            seed=SEED + 2,
        ),
    ]
    return dict(
        frame=scored,
        person_level=person,
        summary=summary,
        summary_by_segment=summary_segment,
        bootstrap=pd.DataFrame(boot),
        top1_agreement=agreement,
        top1_detail=detail,
        n_decisions=len(scored),
        n_people=scored["iid"].nunique(),
    )


def save_outputs(analysis: dict) -> None:
    root = project_root()
    tables = root / TABLES_DIR
    figures = root / FIGURES_DIR
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    lift = analysis["summary"].copy()
    boot = analysis["bootstrap"]
    boot_all = boot[boot["segment"] == "all"]
    for _, row in boot_all.iterrows():
        lift.loc[lift["k"] == row["k"], "lift_ci_low"] = row["ci_low"]
        lift.loc[lift["k"] == row["k"], "lift_ci_high"] = row["ci_high"]
    lift.to_csv(tables / "customer_value_lift.csv", index=False)
    analysis["summary_by_segment"].to_csv(tables / "customer_value_lift_by_segment.csv", index=False)
    analysis["bootstrap"].to_csv(tables / "customer_value_bootstrap.csv", index=False)
    analysis["top1_agreement"].to_csv(tables / "customer_value_top1_agreement.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    all_rows = lift[lift["gender_label"] == "All"]
    x = np.arange(2)
    width = 0.35
    for ax, k in zip(axes, [1, 3]):
        row = all_rows[all_rows["k"] == k].iloc[0]
        vals = [row["stated"], row["revealed"]]
        bars = ax.bar(
            x,
            vals,
            width=0.55,
            color=[STATED_COLOR, REVEALED_COLOR],
            label=["Stated ranking", "Revealed ranking"],
        )
        ax.set_xticks(x, ["Stated weights", "Revealed weights"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Yes rate on top-ranked dates")
        ax.set_title(f"Top {k} date(s) per participant", fontweight="bold")
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=3)
        lift_pp = row["lift_pp"]
        ax.text(
            0.5,
            0.92,
            f"Lift: {lift_pp:+.1%} pp",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
            color="#263747",
        )
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Ranking by revealed preferences raises yes rates on top picks",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.04,
        -0.02,
        f"Same sample as Part 3 ({analysis['n_people']} people, {analysis['n_decisions']:,} dates). "
        "Scores = weighted sum of six 1–10 ratings; weights from *1_1 vs logit revealed shares.",
        fontsize=9,
        color="#526273",
    )
    fig.tight_layout()
    fig.savefig(figures / "customer_value_topk_yes_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    seg = analysis["summary_by_segment"]
    detail = analysis["top1_detail"]
    diff_row = seg[(seg["segment"] == "different_top1") & (seg["gender_label"] == "All") & (seg["k"] == 1)].iloc[0]
    n_diff = int(detail["top1_changed"].sum())
    share_diff = n_diff / len(detail)
    gender_lifts = []
    for code, label in [(0, "Women"), (1, "Men")]:
        sub = seg[
            (seg["segment"] == "different_top1") & (seg["gender_label"] == label) & (seg["k"] == 1)
        ]
        if len(sub):
            gender_lifts.append((label, sub.iloc[0]["lift_pp"]))
    gender_note = " · ".join(f"{lab} {lp:+.0%} pp" for lab, lp in gender_lifts)

    fig2, ax = plt.subplots(figsize=(6, 5))
    vals = [diff_row["stated"], diff_row["revealed"]]
    x = np.arange(2)
    bars = ax.bar(x, vals, width=0.55, color=[STATED_COLOR, REVEALED_COLOR])
    ax.set_xticks(x, ["Stated #1 pick", "Revealed #1 pick"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Yes rate on top-1 date")
    ax.set_title(
        f"When stated and revealed #1 differ (n={n_diff}, ~{share_diff:.0%})",
        fontweight="bold",
    )
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=3)
    ax.text(
        0.5,
        0.92,
        f"Lift: {diff_row['lift_pp']:+.1%} pp",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#263747",
    )
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig2.text(
        0.04,
        -0.04,
        "Each person's stated #1 and revealed #1 are different partners; "
        "yes rate compares those two picks. "
        f"Top-1 lift by gender in this segment: {gender_note}.",
        fontsize=9,
        color="#526273",
    )
    fig2.tight_layout()
    fig2.savefig(figures / "customer_value_topk_different_top1.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    analysis = run_customer_value()
    save_outputs(analysis)
    print(analysis["summary"].round(4).to_string(index=False))
    print("\nLift by top-1 segment (All):")
    seg = analysis["summary_by_segment"]
    print(
        seg[seg["gender_label"] == "All"][["segment", "k", "stated", "revealed", "lift_pp"]]
        .round(4)
        .to_string(index=False)
    )
    print("\nBootstrap lift (pp):")
    print(analysis["bootstrap"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
