"""Run every extended analysis and save its tables, figures and summary menu.

From the repository root:

    python -m src.extended_analysis               # full run
    python -m src.extended_analysis --n-boot 100  # quicker, wider intervals

Part 3 outputs must exist first. Nothing outside the ext_ prefix is written.
Findings and instructions: docs/extended_analysis_report.md.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from . import ext_figures, ext_market, ext_survey
from .constants import SEED, TABLES_DIR
from .ext_features import N_BOOT, bh_adjust, core_sample
from .io import load_clean, project_root

QUESTIONS = {
    "A1": "Can I trust stated-importance surveys?",
    "A2": "Can I write a better survey?",
    "A3": "Should I collect more survey data or more behaviour?",
    "A4": "Do customers understand the market?",
    "A5": "Which metric should I optimise?",
    "A6": "How much can personalisation do?",
    "S1": "Can the say-do gap be shown on one map?",
    "S2": "Does the previous option change how the next is judged?",
    "S3": "Do limited, exclusive recommendations help?",
    "R1": "Is the headline a halo effect?",
    "R2": "Does clipping drive the importance figures?",
    "R3": "Do dropouts bias the sample?",
    "R4": "Does the 1-10 survey format drive the gap?",
    "R5": "Do behavioural segments exist?",
    "R6": "Do survey segments behave differently?",
    "N1": "Do event goals change choices?",
    "N2": "Does similarity matter more to those who say so?",
    "N3": "Does event size shift weight to looks?",
    "N4": "Do polarising looks beat bland ones?",
    "N5": "Can instrumental variables separate rating noise?",
    "N6": "Can the first few choices replace the survey?",
    "N7": "Does one yes make the next less likely?",
    "N8": "For whom does the survey work?",
    "N9": "Does stated importance predict a sensitive choice?",
}
TIERS = {"A": "Core", "S": "Supporting", "R": "Robustness", "N": "Not recommended"}
FIGURES = {
    "A1": "", "A2": "ext_a2_survey_wording.png", "A3": "ext_a3_survey_vs_behaviour.png",
    "A4": "ext_a4_calibration.png; ext_a4_beliefs.png",
    "A5": "ext_a5_yes_vs_match.png; ext_a5_funnel.png", "A6": "ext_a6_variance.png",
    "S1": "ext_s1_perceptual_map.png", "S2": "ext_s2_contrast.png", "S3": "ext_s3_market.png",
    "R1": "ext_robustness.png", "R2": "ext_robustness.png", "R3": "ext_robustness.png",
    "R4": "ext_robustness.png",
}


def run_all(n_boot=N_BOOT, data=None):
    """Run every analysis. Returns tables by name, and the summary menu."""
    data = load_clean() if data is None else data
    ctx = ext_survey.Context(data, core_sample(data), n_boot)
    tables, rows = {}, []
    for analysis in [*ext_survey.ANALYSES, *ext_market.ANALYSES]:
        result = analysis(ctx)
        tables.update(result["tables"])
        rows += result["rows"]
    return tables, build_summary(rows)


def build_summary(rows):
    """One row per finding, with BH-adjusted p across the primary tests."""
    summary = pd.DataFrame(rows)
    summary["p"] = pd.to_numeric(summary["p"], errors="coerce")
    primary = summary["primary"].astype(bool) & summary["p"].notna()
    summary["p_adjusted"] = np.nan
    summary.loc[primary, "p_adjusted"] = bh_adjust(summary.loc[primary, "p"].to_numpy())
    code = summary["id"].str.slice(0, 2)
    summary.insert(1, "tier", code.str[0].map(TIERS))
    summary.insert(2, "question", code.map(QUESTIONS))
    summary["figure"] = code.map(FIGURES).fillna("")
    summary["table"] = summary["table"].map(
        lambda name: name if name.endswith(".csv") else f"ext_{name}.csv")
    order = {"A": 0, "S": 1, "R": 2, "N": 3}
    summary["_rank"] = code.str[0].map(order)
    return (summary.sort_values(["_rank", "id"], kind="stable").drop(columns="_rank")
            .reset_index(drop=True))


def save_tables(tables, summary):
    folder = project_root() / TABLES_DIR
    for name, table in tables.items():
        table.round(6).to_csv(folder / f"ext_{name}.csv", index=False)
    summary.round(6).to_csv(folder / "ext_summary.csv", index=False)


def load_tables():
    """Read saved outputs back, for the notebook."""
    folder = project_root() / TABLES_DIR
    tables = {path.stem[4:]: pd.read_csv(path) for path in folder.glob("ext_*.csv")}
    return tables, tables.pop("summary")


def load_or_run(rebuild=False, n_boot=N_BOOT):
    """Use saved tables unless asked to rebuild or none exist yet."""
    if rebuild or not (project_root() / TABLES_DIR / "ext_summary.csv").exists():
        tables, summary = run_all(n_boot)
        save_tables(tables, summary)
        ext_figures.make_all(*load_tables())
    return load_tables()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    tables, summary = run_all(args.n_boot)
    save_tables(tables, summary)
    ext_figures.make_all(*load_tables())
    columns = ["id", "tier", "estimate", "ci_low", "ci_high", "p", "p_adjusted"]
    print(summary[columns].round(3).to_string(index=False))
    print(f"\nSaved {len(tables) + 1} tables with seed {SEED} and {args.n_boot} bootstrap draws.")


if __name__ == "__main__":
    main()
