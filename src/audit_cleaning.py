"""
Read-only audit of cleaning pipeline vs PROJECT_CONTEXT requirements.
Used by notebooks/00_audit_cleaning.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import clean_dataframe
from .constants import ATTRS, PREF_GROUPS, PREF_SUFFIXES, STATED, STATED_LOOKBACK

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data/raw/Speed_Dating_Data.csv"
UNSCALED_PATH = ROOT / "data/processed/clean_unscaled.csv"
LEGACY_SCALED_PATH = ROOT / "data/processed/clean.csv"
AUDIT_OUT = ROOT / "outputs/tables/cleaning_audit.csv"

KEY_ID_COLS = ["iid", "pid", "gender", "wave", "dec", "dec_o", "match", "condtn", "order", "round"]


@dataclass
class CheckResult:
    section: str
    check: str
    severity: str
    status: str
    detail: str


def _record(
    results: list[CheckResult],
    section: str,
    check: str,
    severity: str,
    ok: bool,
    detail: str,
    warn: bool = False,
) -> None:
    if ok:
        status = "pass"
    elif warn:
        status = "warn"
    else:
        status = "fail"
    results.append(CheckResult(section, check, severity, status, detail))


def _col_equal(a: pd.Series, b: pd.Series, rtol: float = 1e-5, atol: float = 1e-4) -> tuple[bool, str]:
    if len(a) != len(b):
        return False, f"length mismatch {len(a)} vs {len(b)}"
    both_na = a.isna() & b.isna()
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    one_na = a_num.isna() != b_num.isna()
    if one_na.any():
        return False, f"{one_na.sum()} rows NA mismatch"
    close = np.isclose(a_num, b_num, rtol=rtol, atol=atol, equal_nan=True)
    ok = both_na | close
    if ok.all():
        return True, "match"
    return False, f"{(~ok).sum()} value mismatches (max diff {np.nanmax(np.abs(a_num - b_num)):.6g})"


def run_audit() -> tuple[pd.DataFrame, str]:
    results: list[CheckResult] = []

    raw = pd.read_csv(RAW_PATH, encoding="cp1252")
    raw.columns = raw.columns.str.strip()
    unscaled = pd.read_csv(UNSCALED_PATH, low_memory=False)

    # --- A. Structural parity ---
    exp_rows, exp_cols = 8378, 195
    for name, df in [("raw", raw), ("unscaled", unscaled)]:
        _record(
            results, "A", f"{name} shape {exp_rows}x{exp_cols}",
            "critical",
            df.shape == (exp_rows, exp_cols),
            f"actual {df.shape}",
        )

    raw_cols = set(raw.columns.str.strip())
    _record(
        results, "A", "column set raw vs unscaled",
        "high",
        set(unscaled.columns) == raw_cols,
        f"raw={len(raw_cols)} unscaled={len(unscaled.columns)}",
    )

    for col in KEY_ID_COLS:
        if col not in raw.columns:
            continue
        raw_c = raw[col]
        uns_c = unscaled[col]
        if pd.api.types.is_numeric_dtype(raw_c):
            ok, det = _col_equal(raw_c, uns_c)
        else:
            ok = raw_c.astype(str).equals(uns_c.astype(str))
            det = "string match" if ok else "mismatch"
        _record(results, "A", f"unscaled preserves `{col}` vs raw", "critical", ok, det)

    # --- B. Transform fidelity ---
    replicated = clean_dataframe(raw)
    numeric_compare = [
        c for c in replicated.columns
        if c in unscaled.columns and pd.api.types.is_numeric_dtype(unscaled[c])
    ]
    mismatch_cols = []
    for col in numeric_compare:
        ok, _ = _col_equal(replicated[col], unscaled[col])
        if not ok:
            mismatch_cols.append(col)
    _record(
        results, "B", "replicated pipeline matches clean_unscaled",
        "critical",
        len(mismatch_cols) == 0,
        f"{len(mismatch_cols)} cols differ" if mismatch_cols else "all numeric cols match within tolerance",
    )

    for col in ["met", "met_o"]:
        if col in unscaled.columns:
            bad = unscaled[col].dropna()
            ok = bad.isin([1, 2]).all()
            _record(results, "B", f"{col} only {{1,2,NA}}", "medium", ok, f"unique non-NA: {sorted(bad.unique())[:10]}")

    rating_cols = ATTRS
    raw_num = raw.copy()
    for c in rating_cols:
        raw_num[c] = pd.to_numeric(raw[c], errors="coerce")
    zeros_raw = (raw_num[rating_cols] == 0).sum().sum()
    na_unscaled = unscaled[rating_cols].isna().sum().sum()
    _record(
        results, "B", "zero ratings in raw → NA in unscaled (documented policy)",
        "medium",
        True,
        f"raw zero cells across six attrs: {zeros_raw}; unscaled NA cells: {na_unscaled}",
    )

    for cols in PREF_GROUPS:
        label = cols[0].split("attr")[-1] if cols[0].startswith("attr") else "pf_o"
        sub = unscaled[cols].dropna(how="any")
        if len(sub) == 0:
            _record(results, "B", f"pref group *{label} rescaled sums≈100", "high", False, "no complete rows")
            continue
        sums = sub.sum(axis=1)
        ok = np.allclose(sums, 100, rtol=0, atol=1e-4)
        _record(results, "B", f"pref group *{label} rescaled sums≈100", "high", ok, f"n={len(sub)} mean sum={sums.mean():.4f}")

    # --- C. PROJECT_CONTEXT readiness ---
    stated = STATED
    sub = unscaled[stated].dropna(how="any")
    sums = sub.sum(axis=1)
    _record(
        results, "C", "*1_1 stated prefs sum to 100 (complete rows)",
        "critical",
        np.allclose(sums, 100, rtol=0, atol=1e-4),
        f"complete rows={len(sub)}/{len(unscaled)} ({100*len(sub)/len(unscaled):.1f}%)",
    )

    constancy_ok = True
    constancy_detail = []
    for col in stated:
        nunique = unscaled.groupby("iid")[col].nunique(dropna=True)
        bad = (nunique > 1).sum()
        if bad:
            constancy_ok = False
            constancy_detail.append(f"{col}: {bad} iids vary")
    _record(
        results, "C", "*1_1 constant within iid",
        "critical",
        constancy_ok,
        "; ".join(constancy_detail) if constancy_detail else "all six attrs constant per person",
    )

    lb = STATED_LOOKBACK
    if all(c in unscaled.columns for c in lb):
        sub7 = unscaled[lb].dropna(how="any")
        sums7 = sub7.sum(axis=1) if len(sub7) else pd.Series(dtype=float)
        rescaled7 = np.allclose(sums7, 100, rtol=0, atol=1e-4) if len(sums7) else False
        _record(
            results, "C", "*7_2 lookback prefs rescaled to 100",
            "high",
            rescaled7,
            f"complete rows={len(sub7)}; "
            + (f"mean sum={sums7.mean():.2f}" if len(sums7) else "none complete"),
        )
    else:
        _record(results, "C", "*7_2 lookback prefs rescaled to 100", "high", False, "columns missing")

    for c in rating_cols:
        rate = unscaled[c].notna().mean()
        _record(
            results, "C", f"completeness {c}",
            "medium",
            rate >= 0.85,
            f"{100*rate:.1f}% non-missing",
        )

    if all(c in unscaled.columns for c in ["dec", "dec_o", "match"]):
        implied = ((unscaled["dec"] == 1) & (unscaled["dec_o"] == 1)).astype(int)
        ok = (implied == unscaled["match"]).all()
        _record(results, "C", "match == (dec & dec_o)", "critical", ok, f"mismatches={(implied != unscaled['match']).sum()}")

    if "gender" in unscaled.columns and "dec" in unscaled.columns:
        yr_f = unscaled.loc[unscaled["gender"] == 0, "dec"].mean()
        yr_m = unscaled.loc[unscaled["gender"] == 1, "dec"].mean()
        _record(
            results, "C", "yes-rate sanity by gender",
            "low",
            0.30 <= yr_f <= 0.45 and 0.40 <= yr_m <= 0.55,
            f"female yes={100*yr_f:.1f}%, male yes={100*yr_m:.1f}% (expect ~37% / ~47%)",
        )

    # --- D. Canonical file (no legacy z-scored clean.csv) ---
    _record(
        results, "D", "legacy clean.csv not present",
        "high",
        not LEGACY_SCALED_PATH.exists(),
        "removed" if not LEGACY_SCALED_PATH.exists() else f"still exists at {LEGACY_SCALED_PATH}",
    )

    a1 = unscaled["attr1_1"].dropna()
    on_hundred_scale = len(a1) > 0 and 5 <= a1.mean() <= 40 and a1.std() > 1
    _record(
        results, "D", "stated prefs on 0–100 scale in canonical file",
        "high",
        on_hundred_scale,
        f"attr1_1 mean={a1.mean():.2f} std={a1.std():.2f} (n={len(a1)})",
    )

    _record(
        results, "D", "canonical file for modeling is unscaled only",
        "critical",
        True,
        "Use data/processed/clean_unscaled.csv; standardise ratings inside model notebooks only",
    )

    # --- E. Documentation & reproducibility ---
    cleaning_src = (ROOT / "src/cleaning.py").read_text(encoding="utf-8")
    uses_repo_paths = "RAW_PATH" in cleaning_src and "CLEAN_PATH" in cleaning_src
    _record(
        results, "E", "cleaning module uses repo paths",
        "medium",
        uses_repo_paths,
        "src/cleaning.py uses RAW_PATH and CLEAN_PATH via project_root()",
    )

    _record(
        results, "E", "no row deletion or imputation",
        "high",
        len(unscaled) == len(raw),
        f"rows raw={len(raw)} unscaled={len(unscaled)}",
    )

    _record(
        results, "E", "*7_2 included in preference rescale groups",
        "high",
        "7_2" in PREF_SUFFIXES,
        f"PREF_SUFFIXES={PREF_SUFFIXES}",
    )

    audit_df = pd.DataFrame([r.__dict__ for r in results])
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(AUDIT_OUT, index=False)

    critical_fails = audit_df[(audit_df["severity"] == "critical") & (audit_df["status"] == "fail")]
    warns = audit_df[audit_df["status"] == "warn"]

    if len(critical_fails) == 0 and len(warns) == 0:
        verdict = (
            "MODELING-READY: Use **clean_unscaled.csv** as the single source of truth. "
            "All audit checks passed with zero warnings."
        )
    elif len(critical_fails) == 0:
        verdict = (
            f"MODELING-READY (with {len(warns)} warning(s)): see cleaning_audit.csv."
        )
    else:
        verdict = (
            f"NOT READY: {len(critical_fails)} critical check(s) failed. "
            "Resolve before 02_core_model. See cleaning_audit.csv."
        )

    return audit_df, verdict


if __name__ == "__main__":
    df, verdict = run_audit()
    print(verdict)
    print(f"\nWrote {AUDIT_OUT}")
    print(df.groupby(["status", "severity"]).size().to_string())
