"""Data loaders. Always use relative paths from the repo root."""

from pathlib import Path

import pandas as pd

from .constants import CLEAN_PATH, RAW_PATH


def project_root() -> Path:
    """Repo root whether the caller is at root or inside notebooks/."""
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists():
        return cwd
    if (cwd.parent / "data" / "raw").exists():
        return cwd.parent
    raise FileNotFoundError(
        "Could not find data/raw/. Run notebooks from the repo root "
        "or from notebooks/."
    )


def load_raw(path: str | None = None) -> pd.DataFrame:
    """Load the Columbia Speed Dating CSV (encoding cp1252)."""
    root = project_root()
    csv_path = root / (path or RAW_PATH)
    return pd.read_csv(csv_path, encoding="cp1252")


def load_clean(path: str | None = None) -> pd.DataFrame:
    """Load the cleaned file produced by 01_cleaning.ipynb."""
    root = project_root()
    csv_path = root / (path or CLEAN_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run notebooks/01_cleaning.ipynb first."
        )
    return pd.read_csv(csv_path)
