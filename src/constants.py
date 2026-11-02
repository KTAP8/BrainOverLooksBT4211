"""Shared constants used across notebooks."""

SEED = 42

# Six partner attributes (rating columns and preference naming stem)
ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]

# Plain-English labels — always use this order on charts
ATTR_LABELS = [
    "Attractive",
    "Sincere",
    "Intelligent",
    "Fun",
    "Ambitious",
    "Shared interests",
]

# Stated preferences at sign-up (self-explicated, 100-point allocation)
STATED = [a + "1_1" for a in ATTRS]

# Looking-back stated preferences (day after) — for self-correction analysis
STATED_LOOKBACK = [a + "7_2" for a in ATTRS]

# 100-point preference groups to rescale during cleaning
PREF_SUFFIXES = ["1_1", "4_1", "2_1", "1_2", "7_2"]

PREF_GROUPS = [[f"{p}{s}" for p in ATTRS] for s in PREF_SUFFIXES] + [
    ["pf_o_att", "pf_o_sin", "pf_o_int", "pf_o_fun", "pf_o_amb", "pf_o_sha"]
]

# Never use as predictors of `dec`
LEAKY = ["match", "dec_o", "like", "prob"]

# Chart colours — keep fixed across all slides
STATED_COLOR = "#4C78A8"    # blue
REVEALED_COLOR = "#F58518"  # orange

RAW_PATH = "data/raw/Speed_Dating_Data.csv"
CLEAN_PATH = "data/processed/clean_unscaled.csv"
FIGURES_DIR = "outputs/figures"
TABLES_DIR = "outputs/tables"
