"""Charts for the extended analyses, read from the saved ext_ tables.

Stated and revealed keep the project's fixed colours; anything else is neutral
grey, so colour never means anything new. Bars and points are labelled directly.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors  # noqa: F401  (used through matplotlib.colors)
import matplotlib.ticker  # noqa: F401  (used through matplotlib.ticker)
import numpy as np

from .constants import FIGURES_DIR, REVEALED_COLOR, STATED_COLOR
from .io import project_root

INK = "#263747"
MUTED = "#526273"
GRID = "#E3E7EB"
NEUTRAL = "#8A94A0"
LIGHT = "#C7CDD3"
SURFACE = "#FFFFFF"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlesize": 12, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "text.color": INK, "axes.grid": False})


def _save(fig, name):
    fig.savefig(project_root() / FIGURES_DIR / name, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _title(fig, title, note=None, top=1.02):
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK, y=top)
    if note:
        fig.text(0.02, -0.03 - 0.28 / fig.get_figheight(), note, fontsize=9, color=MUTED,
                 ha="left", va="top")


def _hgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def survey_wording(tables):
    wording = tables["a2_wording"]
    order = ["1_1", "1_2", "4_1", "2_1"]
    labels = ["What I look for\n(sign-up)", "What I look for\n(day after)",
              "What others of my\ngender look for", "What the opposite\nsex looks for"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, gender in zip(axes, ("Women", "Men")):
        side = wording[wording["gender"] == gender].set_index("suffix").loc[order]
        y = np.arange(len(order))[::-1]
        for yi, (_, row) in zip(y, side.iterrows()):
            ax.plot([row["attractive_share"], row["attractive_target"]], [yi, yi], color=LIGHT, lw=2, zorder=1)
            ax.text((row["attractive_share"] + row["attractive_target"]) / 2, yi + 0.17,
                    f"gap {row['attractive_gap']:.1f}", ha="center", fontsize=9, color=MUTED)
        ax.scatter(side["attractive_share"], y, s=80, color=STATED_COLOR, zorder=3, label="Survey answer")
        ax.scatter(side["attractive_target"], y, s=80, color=REVEALED_COLOR, zorder=3, label="Actual choices")
        ax.set_yticks(y, labels)
        ax.set_xlim(10, 52)
        ax.set_ylim(-0.6, 3.6)
        ax.set_title(f"{gender} answering", loc="left")
        ax.set_xlabel("Weight on attractiveness (out of 100)")
        _hgrid(ax)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9, ncol=2, bbox_to_anchor=(0, 1.08))
    _title(fig, "Asking what others want lands closer to real choices",
           "Waves 10-21, same people. 'Opposite sex' is compared with that group's choices.", top=1.13)
    _save(fig, "ext_a2_survey_wording.png")


def survey_vs_behaviour(tables):
    table = tables["a3_prediction"]
    order = ["Hobbies", "Demographics", "Self-ratings", "Stated weights", "Lifestyle",
             "Full sign-up survey", "First 3 speed dates: yeses only",
             "First 3 speed dates: ratings and yeses"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, sharex=True)
    for ax, gender in zip(axes, ("Women", "Men")):
        side = table[table["people_rated"] == gender].set_index("predictors").loc[order]
        values = side["r2_ridge"].clip(lower=0).to_numpy()
        colours = [REVEALED_COLOR if "speed dates" in n else STATED_COLOR for n in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, values, color=colours, height=0.62)
        for yi, v, raw in zip(y, values, side["r2_ridge"]):
            ax.text(v + 0.006, yi, f"{raw:.2f}".replace("-0.00", "0.00"), va="center", fontsize=9, color=INK)
        ax.set_yticks(y, order)
        ax.set_title(f"Predicting yeses for {gender.lower()}", loc="left")
        ax.set_xlabel("Variance explained on unseen events (R²)")
        _hgrid(ax)
    _title(fig, "Three speed dates predict demand better than the whole sign-up survey",
           "Ridge regression, tested on held-out events. Negative R² is drawn as zero and labelled.")
    _save(fig, "ext_a3_survey_vs_behaviour.png")


def calibration(tables):
    bins = tables["a4_calibration_bins"]
    labels = ["1-2", "3-4", "5-6", "7-8", "9-10"]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(x, [0.15, 0.35, 0.55, 0.75, 0.95], color=LIGHT, lw=2, ls="--", zorder=1)
    ax.plot(x, bins["i_said_yes"], color=STATED_COLOR, lw=2, marker="o", ms=8, zorder=2)
    ax.plot(x, bins["partner_said_yes"], color=REVEALED_COLOR, lw=2, marker="o", ms=8, zorder=3)
    ax.text(x[-1] + 0.14, bins["i_said_yes"].iloc[-1] + 0.025, "I said yes", va="center", fontsize=10)
    ax.text(x[-1] + 0.14, bins["partner_said_yes"].iloc[-1] - 0.035, "Partner said yes", va="center", fontsize=10)
    ax.text(x[-1] + 0.08, 0.95, "If guesses\nwere right", va="center", fontsize=10, color=MUTED)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.2, 5.3)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("My guess that the partner will say yes (1-10)")
    ax.set_ylabel("Share of dates")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _title(fig, "People's guesses track their own feelings more than their partner's")
    _save(fig, "ext_a4_calibration.png")


def beliefs(tables):
    table = tables["a4_beliefs"]
    order = ["Attractive", "Sincere", "Intelligent", "Fun", "Ambitious", "Shared interests"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, sharex=True)
    for ax, believer in zip(axes, ("Women", "Men")):
        side = table[table["believer"] == believer].set_index("attribute").loc[order]
        y = np.arange(len(order))[::-1]
        for yi, (_, row) in zip(y, side.iterrows()):
            ax.plot([row["believed"], row["choosers_actual"]], [yi, yi], color=LIGHT, lw=2, zorder=1)
        ax.scatter(side["believed"], y, s=80, color=STATED_COLOR, zorder=3, label="Believed")
        ax.scatter(side["choosers_actual"], y, s=80, color=REVEALED_COLOR, zorder=3, label="Actual choices")
        ax.set_yticks(y, order)
        other = "men" if believer == "Women" else "women"
        ax.set_title(f"What {believer.lower()} think {other} want", loc="left")
        ax.set_xlabel("Weight (out of 100)")
        _hgrid(ax)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9, ncol=2, bbox_to_anchor=(0, 1.07))
    _title(fig, "Both genders underrate shared interests and overrate ambition",
           "Waves 10-21. Actual choices are that group's revealed importance.", top=1.13)
    _save(fig, "ext_a4_beliefs.png")


def yes_vs_match(tables):
    table = tables["a5_ranking"]
    order = ["Random order", "Stated weights", "Revealed weights", "Both likely to say yes"]
    side = table[table["top_k"] == 3].set_index("rule").loc[order]
    colours = [LIGHT, STATED_COLOR, REVEALED_COLOR, NEUTRAL]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, column, title in zip(axes, ("yes_rate", "match_rate"), ("Yes rate", "Mutual match rate")):
        y = np.arange(len(order))[::-1]
        ax.barh(y, side[column], color=colours, height=0.62)
        for yi, v in zip(y, side[column]):
            ax.text(v + 0.012, yi, f"{v:.0%}", va="center", fontsize=10)
        ax.set_yticks(y, order)
        ax.set_xlim(0, 0.85 if column == "yes_rate" else 0.45)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.set_title(f"{title}, top 3 picks", loc="left")
        _hgrid(ax)
    _title(fig, "Only ranking for mutual interest clearly raises matches",
           "Pre-date information only, tested on events the models never saw.")
    _save(fig, "ext_a5_yes_vs_match.png")


def funnel(tables):
    table = tables["a5_funnel"].set_index("stage")
    stages = [("Speed dates per person", "Speed dates"), ("Yeses given", "Yeses given"),
              ("Mutual matches", "Mutual matches")]
    values = [table.loc[s, "value"] for s, _ in stages]
    fig, ax = plt.subplots(figsize=(7.4, 3))
    y = np.arange(3)[::-1]
    ax.barh(y, values, color=[LIGHT, NEUTRAL, REVEALED_COLOR], height=0.62)
    for yi, v in zip(y, values):
        ax.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=11)
    ax.set_yticks(y, [label for _, label in stages])
    ax.set_xlabel("Average per person")
    _hgrid(ax)
    _title(fig, "Most interest never becomes a mutual match",
           f"{table.loc['Share with no match', 'value']:.0%} of people get no match. "
           f"{table.loc['Went on a date with a match', 'value']:.0%} of follow-up respondents "
           f"reported a date (n={int(table.loc['Went on a date with a match', 'people'])}).")
    _save(fig, "ext_a5_funnel.png")


def variance(tables):
    table = tables["a6_variance"]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    y = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    parts = [("chooser_share", "The chooser", STATED_COLOR), ("partner_share", "The partner", REVEALED_COLOR),
             ("unexplained_share", "Unexplained", LIGHT)]
    for column, label, colour in parts:
        values = table[column].to_numpy() * 100
        ax.barh(y, values - 0.4, left=left + 0.2, color=colour, height=0.6, label=label)
        for yi, l, v in zip(y, left, values):
            ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=10,
                    color="white" if colour == STATED_COLOR else INK)
        left += values
    ax.set_yticks(y, table["group"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of variation in yes decisions")
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    _hgrid(ax)
    _title(fig, "Most of a yes can't be traced to who is choosing or who is chosen",
           "Unexplained includes chemistry and random noise; the two can't be separated here.", top=1.14)
    _save(fig, "ext_a6_variance.png")


def perceptual_map(tables):
    people, arrows = tables["s1_people"], tables["s1_arrows"]
    axes_table = tables["s1_axes"]
    shades = matplotlib.colors.LinearSegmentedColormap.from_list("shares", [GRID, INK])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    for ax, code, gender in zip(axes, (1, 0), ("Women", "Men")):
        rated = people[people["gender"] == code]
        dots = ax.scatter(rated["axis_appeal"], rated["axis_brains"], c=rated["yes_received"],
                          cmap=shades, vmin=0, vmax=1, s=16, linewidths=0, alpha=0.85)
        side = arrows[arrows["gender"] == gender].set_index("source")
        scale = 0.09
        for source, colour in (("Stated", STATED_COLOR), ("Revealed", REVEALED_COLOR)):
            dx, dy = side.loc[source, "appeal"] * scale, side.loc[source, "brains"] * scale
            ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", lw=2.4, color=colour, mutation_scale=16))
            ax.text(dx + 0.25, dy, source, color=INK, fontsize=10, fontweight="bold", va="center",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.9), zorder=5)
        ax.set_ylim(-3.9, 4.7)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_title(f"{gender} choosing: {'men' if gender == 'Women' else 'women'} as dots", loc="left")
        ax.set_xlabel(f"Overall appeal ({axes_table['variance_share'][0]:.0%} of variance)")
    axes[0].set_ylabel(f"Looks  ←  → Brains ({axes_table['variance_share'][1]:.0%} of variance)")
    colourbar = fig.colorbar(dots, ax=axes, fraction=0.025, pad=0.02)
    colourbar.set_label("Share of partners saying yes")
    _title(fig, "Women say brains and choose looks; men choose looks more than they say",
           "Each dot is a person, placed by ratings received. Arrows show preference direction, scaled to fit.")
    _save(fig, "ext_s1_perceptual_map.png")


def contrast(tables):
    table = tables["s2_contrast"]
    row = table[(table["group"] == "Everyone")
                & (table["effect"] == "Previous partner's attractiveness")].iloc[0]
    draws = tables["s2_placebo"]["shuffled_estimate_pp"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.hist(draws, bins=30, color=LIGHT, edgecolor=SURFACE)
    ax.axvline(row["estimate_pp"], color=REVEALED_COLOR, lw=2.4)
    ax.text(row["estimate_pp"] + 0.05, ax.get_ylim()[1] * 0.9,
            f"Real order: {row['estimate_pp']:.1f} pp", fontsize=10)
    ax.text(0.45, ax.get_ylim()[1] * 0.72, "Shuffled order", fontsize=10, color=MUTED)
    ax.set_xlabel("Change in next yes per 1 SD more attractive previous partner (pp)")
    ax.set_ylabel("Shuffles")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _title(fig, "An attractive previous partner lowers the next yes")
    _save(fig, "ext_s2_contrast.png")


def market(tables):
    table = tables["s3_market"]
    one = table[table["slots"] == 1].iloc[0]
    order = [("random", "Random", LIGHT), ("revealed", "Stable matching, revealed", REVEALED_COLOR),
             ("stated", "Stable matching, stated", STATED_COLOR), ("optimised", "Optimised for both", NEUTRAL)]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    y = np.arange(4)[::-1]
    ax.barh(y, [one[k] for k, _, _ in order], color=[c for _, _, c in order], height=0.62)
    for yi, (key, _, _) in zip(y, order):
        ax.text(one[key] + 0.004, yi, f"{one[key]:.1%}", va="center", fontsize=10)
    ax.set_yticks(y, [label for _, label, _ in order])
    ax.set_xlim(0, 0.31)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Pairs who both said yes")
    _hgrid(ax)
    _title(fig, "Optimised pairing beats random; the structured rules are too close to separate",
           "One exclusive recommendation each, later-evening pairs only, 21 events.")
    _save(fig, "ext_s3_market.png")


def robustness(tables):
    published = tables["a1_gaps"]
    consensus = tables["r1_consensus"]
    relative = tables["r2_relative_weights"]
    weighted = tables["r3_dropout_weights"]
    trimmed = tables["r4_excluding_rating_waves"]

    def pick(frame, gender, column):
        return float(frame[(frame["gender"] == gender) & (frame.get("attribute", frame.get("label")) == "Attractive")][column].iloc[0])

    methods = [("Published model", published, "revealed"), ("Dropouts reweighted", weighted, "reweighted"),
               ("100-point waves only", trimmed, "revealed"), ("No clipping", relative, "relative_weight"),
               ("Other people's ratings", consensus, "revealed_consensus")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True, sharex=True)
    for ax, gender in zip(axes, ("Women", "Men")):
        stated = pick(published, gender, "stated")
        values = [pick(frame, gender, column) for _, frame, column in methods]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, values, color=[REVEALED_COLOR] + [NEUTRAL] * 4, height=0.62)
        ax.axvline(stated, color=STATED_COLOR, lw=2)
        ax.text(stated + 0.8, len(methods) - 0.42, f"Stated {stated:.1f}", fontsize=9)
        for yi, v in zip(y, values):
            ax.text(v + 0.8, yi, f"{v:.1f}", va="center", fontsize=10)
        ax.set_yticks(y, [label for label, _, _ in methods])
        ax.set_xlim(0, 80)
        ax.set_ylim(-0.6, len(methods) - 0.1)
        ax.set_title(gender, loc="left")
        ax.set_xlabel("Weight on attractiveness")
        _hgrid(ax)
    _title(fig, "However it's measured, attractiveness outweighs what people state")
    _save(fig, "ext_robustness.png")


def make_all(tables, summary=None):
    _style()
    for chart in (survey_wording, survey_vs_behaviour, calibration, beliefs, yes_vs_match, funnel,
                  variance, perceptual_map, contrast, market, robustness):
        chart(tables)
