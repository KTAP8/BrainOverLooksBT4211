"""Final images for the two "rank for both sides" slides (A5o version).

deck_full_8c_model_final.png  How "rank for both sides" works: one model run both ways.
deck_full_8c_race_final.png   Top-3 match rate by ranking rule, both sides = A5o same model.

Same palette and layout as deck_visuals.slide_8c_model / slide_8c_race, with the
model card text, inputs and numbers changed for A5o. Numbers come from the tables
only: ext_a5_ranking.csv, ext_a5_direct_vs_two_sided.csv, ext_a5_seed_sensitivity.csv
and ext_a5_same_model.csv (run src/ext_same_model.py first).

Usage: python src/deck_final_8c.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
CREAM, RUST, SAND, ORANGE, MUTED = "#fff3dd", "#7c332f", "#d8c3a0", "#F58518", "#c9a99a"
MAROON, DEEP = "#4c0f12", "#3a0b0e"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
SAME = "Same model both ways"


def table(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def pct(v):
    return f"{100 * v:.0f}%"


def blend(colour, background, amount):
    a, b = mc.to_rgb(colour), mc.to_rgb(background)
    return tuple(amount * x + (1 - amount) * y for x, y in zip(a, b))


def slide():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis("off")
    fig.patch.set_facecolor(MAROON)
    return fig, ax


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"deck_full_{name}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def card(ax, x, y, w, h, fill, edge, lw, radius):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=lw))


def arrow(ax, x1, y1, x2, y2, colour=CREAM, lw=3, head=18):
    ax.plot([x1, x2], [y1, y2], color=colour, linewidth=lw, solid_capstyle="round")
    angle = np.arctan2(y2 - y1, x2 - x1)
    left = (x2 - head * np.cos(angle - 0.45), y2 - head * np.sin(angle - 0.45))
    right = (x2 - head * np.cos(angle + 0.45), y2 - head * np.sin(angle + 0.45))
    ax.add_patch(Polygon([(x2, y2), left, right], facecolor=colour, edgecolor="none"))


def model_slide():
    fig, ax = slide()
    card(ax, 60, 300, 470, 700, DEEP, MUTED, 1.5, 26)
    ax.text(100, 355, "EACH PROFILE", color=MUTED, fontsize=19, fontweight="bold", va="center")
    ax.text(100, 395, "known before you meet", color=MUTED, fontsize=16, va="center")
    for i, text in enumerate(["How others rated them", "How others rated you", "Your yes rate",
                              "Their yes rate", "Age gap, hobbies, race"]):
        ax.text(100, 490 + i * 112, text, color=CREAM, fontsize=23, va="center")
    rose = blend(RUST, CREAM, 0.85)
    for top, colour, tag, question, lines in (
            (300, SAND, "A", "Will YOU say yes?",
             ["Shared weights × how others rated THEM", "+ pair facts, scaled by YOUR yes rate"]),
            (690, rose, "B", "Will THEY say yes?",
             ["Shared weights × how others rated YOU", "+ pair facts, scaled by THEIR yes rate"])):
        card(ax, 640, top, 720, 310, DEEP, colour, 3, 28)
        ax.add_patch(Circle((710, top + 75), 38, facecolor=colour, edgecolor="none"))
        ax.text(710, top + 76, tag, color=MAROON, fontsize=30, fontweight="bold", ha="center", va="center")
        ax.text(770, top + 75, question, color=CREAM, fontsize=34, fontweight="bold", va="center")
        ax.text(680, top + 175, lines[0], color=CREAM, fontsize=24, va="center")
        ax.text(680, top + 235, lines[1], color=CREAM, fontsize=24, va="center")
    ax.text(1000, 650, "SAME MODEL, RUN BOTH WAYS", color=MUTED, fontsize=17, fontweight="bold",
            ha="center", va="center")
    arrow(ax, 545, 560, 625, 470)
    arrow(ax, 545, 740, 625, 830)
    arrow(ax, 1370, 455, 1440, 600, colour=ORANGE, lw=5, head=20)
    arrow(ax, 1370, 845, 1440, 700, colour=ORANGE, lw=5, head=20)
    card(ax, 1450, 560, 270, 180, ORANGE, CREAM, 3, 28)
    ax.text(1585, 625, "P(A) × P(B)", color=MAROON, fontsize=30, fontweight="bold", ha="center", va="center")
    ax.text(1585, 690, "chance of a match", color=MAROON, fontsize=17, ha="center", va="center")
    arrow(ax, 1728, 650, 1768, 650, colour=ORANGE, lw=5, head=16)
    ax.text(1820, 505, "TOP 3", color=MUTED, fontsize=17, fontweight="bold", ha="center", va="center")
    for i in range(3):
        y = 540 + i * 82
        card(ax, 1775, y, 110, 66, DEEP if i else ORANGE, CREAM, 1.5, 16)
        ax.add_patch(Circle((1808, y + 33), 18, facecolor=CREAM if i else MAROON, edgecolor="none"))
        ax.text(1808, y + 34, str(i + 1), color=MAROON if i else CREAM, fontsize=15, fontweight="bold",
                ha="center", va="center")
        ax.plot([1836, 1868], [y + 33, y + 33], color=CREAM if i else MAROON, linewidth=5, alpha=0.6,
                solid_capstyle="round")
    save(fig, "8c_model_final")


def race_slide():
    ranking = table("ext_a5_ranking")
    top3 = ranking[ranking.top_k == 3].set_index("rule")
    direct = table("ext_a5_direct_vs_two_sided")
    direct = direct[(direct.top_k == 3) & (direct.metric.isin(["Match rate", "Yes rate"]))]
    same = table("ext_a5_same_model")
    same = same[same["rule"] == SAME].set_index("seed")
    rates = {rule: (top3.loc[rule, "match_rate"], top3.loc[rule, "yes_rate"]) for rule in top3.index}
    main_direct = direct[direct.seed == 42].set_index("metric")["direct_model"]
    rates["direct"] = (main_direct["Match rate"], main_direct["Yes rate"])
    rates["same"] = (same.loc[42, "top3_match"], same.loc[42, "top3_yes"])
    direct_match = direct[direct.metric == "Match rate"].set_index("seed")["direct_model"]
    vs_direct = (same["top3_match"] - direct_match.loc[same.index]) * 100
    low, high = same["top3_match"].min(), same["top3_match"].max()
    yes_cost = (top3.loc["Stated weights", "yes_rate"] - same.loc[42, "top3_yes"]) * 100
    lanes = [("Random order", "Random order", MUTED), ("Survey weights", "Stated weights", SAND),
             ("Behaviour weights", "Revealed weights", SAND),
             ("Optimise for matches", "direct", blend(ORANGE, MAROON, 0.45)),
             ("Rank for both sides", "same", ORANGE)]
    fig, ax = slide()
    x0, scale, step, height = 470, 20.0, 124, 104
    ax.text(60, 215, "MATCH RATE IN EACH PERSON'S TOP 3", color=MUTED, fontsize=16, fontweight="bold",
            va="center")
    for i, (label, rule, colour) in enumerate(lanes):
        y = 250 + i * step
        mid = y + height / 2
        ax.add_patch(FancyBboxPatch((40, y), 1300, height, boxstyle="round,pad=0,rounding_size=20",
                                    facecolor=DEEP, edgecolor="none"))
        ax.text(70, mid - 17, label, color=CREAM, fontsize=20,
                fontweight="bold" if colour == ORANGE else "normal", va="center")
        value, yes = rates[rule]
        prefix = "one model  ·  " if rule == "direct" else ""
        ax.text(70, mid + 20, f"{prefix}yes rate {pct(yes)}", color=MUTED, fontsize=14, va="center")
        end = x0 + value * 100 * scale
        ax.add_patch(FancyBboxPatch((x0, mid - 26), end - x0, 52, boxstyle="round,pad=0,rounding_size=26",
                                    facecolor=colour, edgecolor="none"))
        ax.add_patch(Circle((end - 26, mid), 16, facecolor=MAROON, edgecolor="none"))
        ax.text(end + 16, mid, pct(value), color=ORANGE if colour == ORANGE else CREAM, fontsize=28,
                fontweight="bold", va="center")
        if colour == ORANGE:
            ax.plot([end + 100, end + 100], [y + 8, y + height - 8], color=CREAM, linewidth=3)
            ax.add_patch(Polygon([(end + 100, y + 8), (end + 152, y + 25), (end + 100, y + 42)],
                                 facecolor=ORANGE))
    bottom = 250 + len(lanes) * step
    for x in range(int(x0), int(x0 + 40 * scale) + 1, int(10 * scale)):
        ax.plot([x, x], [240, bottom], color=CREAM, alpha=0.12, linewidth=1.5, linestyle=(0, (6, 8)))
    ax.text(60, 1035, f"Unseen events, one random split  ·  both sides: {100 * low:.0f}–{100 * high:.0f}% "
            f"across 5 splits  ·  beats one match model by +{vs_direct.min():.0f} to +{vs_direct.max():.0f} pts"
            f"  ·  costs ~{yes_cost:.0f} pts of yeses", color=MUTED, fontsize=14, va="center")
    save(fig, "8c_race_final")
    print({k: round(v[0], 3) for k, v in rates.items() if k in ("Stated weights", "direct", "same")},
          "range", round(low, 3), round(high, 3), "vs direct", vs_direct.round(1).to_dict(),
          "yes cost", round(yes_cost, 1))


if __name__ == "__main__":
    model_slide()
    race_slide()
