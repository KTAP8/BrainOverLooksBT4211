"""Speed dating data cleaning pipeline (single source of truth)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import ATTRS, CLEAN_PATH, PREF_GROUPS, RAW_PATH
from .io import project_root

ATTRIBUTE_PREFIXES = ("attr", "sinc", "intel", "fun", "amb", "shar")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning transforms; does not read or write files."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    df["met"] = df["met"].replace(0, 2)
    for col in ["met", "met_o"]:
        if col in df.columns:
            df[col] = df[col].where(df[col].isin([1, 2]))

    numeric_cols = [
        "age", "age_o", "mn_sat", "tuition", "income",
        "round", "order", "int_corr", "imprace", "imprelig",
        "sports", "tvsports", "exercise", "dining", "museums",
        "art", "hiking", "gaming", "clubbing", "reading",
        "tv", "theater", "movies", "concerts", "music",
        "shopping", "yoga", "exphappy", "expnum",
        "like", "prob", "like_o", "prob_o", "match_es",
        "satis_2", "you_call", "them_cal", "numdat_3", "num_in_3",
    ]
    numeric_cols += [
        col for col in df.columns
        if col.startswith(ATTRIBUTE_PREFIXES) or col.startswith("pf_o_")
    ]
    numeric_cols = list(dict.fromkeys(c for c in numeric_cols if c in df.columns))

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col].astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        )

    rating_cols = [
        "imprace", "imprelig",
        "sports", "tvsports", "exercise", "dining", "museums",
        "art", "hiking", "gaming", "clubbing", "reading",
        "tv", "theater", "movies", "concerts", "music",
        "shopping", "yoga", "exphappy", "satis_2",
        "attr", "sinc", "intel", "fun", "amb", "shar", "like", "prob",
        "attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o", "like_o", "prob_o",
    ]
    for suffix in ["3_1", "5_1", "3_s", "3_2", "5_2", "3_3", "5_3"]:
        rating_cols.extend(f"{prefix}{suffix}" for prefix in ["attr", "sinc", "intel", "fun", "amb"])

    for col in dict.fromkeys(rating_cols):
        if col in df.columns:
            df[col] = df[col].where(df[col].between(1, 10))

    for cols in PREF_GROUPS:
        values = df[cols]
        totals = values.sum(axis=1, min_count=6)
        valid = (
            values.notna().all(axis=1)
            & values.ge(0).all(axis=1)
            & values.le(100).all(axis=1)
            & totals.gt(0)
        )
        df[cols] = values.div(totals.where(valid), axis=0) * 100

    return df


def run_cleaning(
    raw_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw CSV, clean, optionally write canonical unscaled output."""
    root = project_root()
    raw_csv = root / (raw_path or RAW_PATH)
    out_csv = root / (out_path or CLEAN_PATH)

    df = pd.read_csv(raw_csv, encoding="cp1252")
    cleaned = clean_dataframe(df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_csv, index=False)
    return cleaned


if __name__ == "__main__":
    result = run_cleaning()
    print(f"Wrote {len(result)} rows to {CLEAN_PATH}")
