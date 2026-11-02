"""CLI entry point for cleaning (delegates to src.cleaning)."""

from .cleaning import run_cleaning

if __name__ == "__main__":
    df = run_cleaning()
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).to_string())
