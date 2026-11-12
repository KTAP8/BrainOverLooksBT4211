"""Deck-styled visuals for Person 6's slides (Canva pages 19-24).

Transparent PNGs in the team deck's colours, written to
outputs/figures/deck_*.png. Every number is read from outputs/tables, so the
slides stay in step with the analysis. The repo's own charts are unchanged;
these use sand for "survey" instead of the repo blue, because blue clashes
with the deck's maroon. Run: python -m src.deck_visuals
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge  # noqa: E402

from .io import project_root  # noqa: E402

CREAM, RUST, SAND, ORANGE, MUTED = "#fff3dd", "#7c332f", "#d8c3a0", "#F58518", "#c9a99a"
ARROW = r"$\rightarrow$"
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
TABLES = project_root() / "outputs" / "tables"
FIGURES = project_root() / "outputs" / "figures"


def table(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def canvas(width_px, height_px):
    """Figure in slide pixels (1920x1080 deck), drawn on a pixel grid."""
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(FIGURES / f"deck_{name}.png", dpi=300, transparent=True)
    plt.close(fig)


def card(ax, x, y, w, h, fill=RUST, edge="#ffffff", lw=1.5, radius=28):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, linewidth=lw))


def pct(value):
    return f"{100 * value:.0f}%"


# ------------------------------------------------------------------ slides


def ranking_donut():
    """8a: why the pooled lift is small."""
    row = table("customer_value_top1_agreement").set_index("segment").loc["all"]
    same = row["share_same_top1"]
    fig, ax = canvas(1124, 640)
    cx, cy, r = 300, 330, 230
    ax.add_patch(Wedge((cx, cy), r, 90, 90 + 360 * same, width=70, facecolor=SAND, edgecolor="none"))
    ax.add_patch(Wedge((cx, cy), r, 90 + 360 * same + 1.5, 450 - 1.5, width=70, facecolor=ORANGE,
                       edgecolor="none"))
    ax.text(cx, cy - 15, "1 in 3", color=ORANGE, fontsize=58, fontweight="bold", ha="center",
            va="center")
    ax.text(cx, cy + 50, "see a different #1", color=CREAM, fontsize=22, ha="center", va="center")
    ax.text(600, 190, pct(same), color=SAND, fontsize=40, fontweight="bold", va="center")
    ax.text(600, 240, "Same #1 either way.\nNo change for them.", color=CREAM, fontsize=21, va="top",
            linespacing=1.4)
    ax.text(600, 380, pct(1 - same), color=ORANGE, fontsize=40, fontweight="bold", va="center")
    ax.text(600, 430, "#1 changes.\nThat's where the lift lives.", color=CREAM, fontsize=21, va="top",
            linespacing=1.4)
    ax.text(40, 610, f"Top-pick yes rate {pct(row['stated_top1_yes_rate'])} {ARROW} "
            f"{pct(row['revealed_top1_yes_rate'])}  ·  best case: ratings made after the date",
            color=MUTED, fontsize=15, va="center")
    save(fig, "8a_donut")


def phone(ax, x, label, rate, colour, highlight):
    w, h = 380, 470
    card(ax, x, 40, w, h, fill="#5e1f1f" if not highlight else RUST,
         edge=colour if highlight else "#ffffff", lw=4 if highlight else 1.5, radius=46)
    ax.add_patch(FancyBboxPatch((x + w / 2 - 50, 62), 100, 14, boxstyle="round,pad=0,rounding_size=7",
                                facecolor="#ffffff", alpha=0.35, edgecolor="none"))
    ax.add_patch(Circle((x + w / 2, 170), 62, facecolor=colour, alpha=0.9, edgecolor="none"))
    ax.add_patch(Circle((x + w / 2, 150), 24, facecolor=CREAM, edgecolor="none"))
    ax.add_patch(Wedge((x + w / 2, 222), 42, 180, 360, facecolor=CREAM, edgecolor="none"))
    ax.text(x + w / 2, 270, label, color=CREAM, fontsize=20, ha="center", va="center")
    ax.text(x + w / 2, 360, pct(rate), color=colour, fontsize=72, fontweight="bold", ha="center",
            va="center")
    ax.text(x + w / 2, 430, "said yes", color=CREAM, fontsize=22, ha="center", va="center")


def twist_phones():
    """8b: when the two rules disagree, whose pick gets the yes?"""
    row = table("customer_value_top1_agreement").set_index("segment").loc["different_top1"]
    boot = table("customer_value_bootstrap")
    ci = boot[(boot.segment == "different_top1") & (boot.k == 1)].iloc[0]
    fig, ax = canvas(1124, 640)
    phone(ax, 110, "Survey's #1 pick", row["stated_top1_yes_rate"], SAND, False)
    phone(ax, 634, "Behaviour's #1 pick", row["revealed_top1_yes_rate"], ORANGE, True)
    ax.text(562, 275, "vs", color=CREAM, fontsize=30, style="italic", ha="center", va="center")
    ax.text(40, 600, f"{int(row['n_people'])} people whose #1 differs  ·  95% CI "
            f"+{100 * ci['ci_low']:.0f} to +{100 * ci['ci_high']:.0f} pts",
            color=MUTED, fontsize=15, va="center")
    save(fig, "8b_phones")


def match_bars():
    """8c: ranking for both people raises matches."""
    ranking = table("ext_a5_ranking")
    top3 = ranking[ranking.top_k == 3].set_index("rule")
    rows = [("Random order", "Random order", MUTED), ("Survey weights", "Stated weights", SAND),
            ("Behaviour weights", "Revealed weights", SAND),
            ("Both likely to say yes", "Both likely to say yes", ORANGE)]
    benefit = table("ext_a5_who_benefits").set_index(["tier", "rule"])
    before = benefit.loc[("Least chosen third", "Stated weights"), "no_match_share"]
    after = benefit.loc[("Least chosen third", "Both likely to say yes"), "no_match_share"]
    fig, ax = canvas(1124, 640)
    ax.text(40, 50, "Match rate on each person's top 3 picks", color=CREAM, fontsize=22,
            fontweight="bold", va="center")
    scale = 1300
    for i, (label, rule, colour) in enumerate(rows):
        y = 120 + i * 82
        value = top3.loc[rule, "match_rate"]
        ax.text(40, y + 26, label, color=CREAM, fontsize=19, va="center")
        ax.add_patch(FancyBboxPatch((300, y), value * scale, 52, boxstyle="round,pad=0,rounding_size=6",
                                    facecolor=colour, edgecolor="none"))
        ax.text(300 + value * scale + 16, y + 26, pct(value), color=colour if colour == ORANGE else CREAM,
                fontsize=24, fontweight="bold", va="center")
    card(ax, 40, 460, 1044, 100, fill="#5e1f1f", edge=ORANGE, lw=2.5, radius=22)
    ax.text(70, 510, "Least-chosen users with no match in their top 3:", color=CREAM, fontsize=20,
            va="center")
    ax.text(1050, 510, f"{pct(before)} {ARROW} {pct(after)}", color=ORANGE, fontsize=30, fontweight="bold",
            ha="right", va="center")
    ax.text(40, 605, "Pre-date info, unseen events  ·  +2.5 to +7.1 pts across tests  ·  "
            "costs 3–5 pts of yeses  ·  cap mass-likers", color=MUTED, fontsize=15, va="center")
    save(fig, "8c_matches")


def survey_vs_dates():
    """8d: three speed dates beat the whole sign-up survey."""
    pred = table("ext_a3_prediction")
    survey = pred[pred.predictors == "Full sign-up survey"]["r2_ridge"]
    dates = pred[pred.predictors == "First 3 speed dates: ratings and yeses"]["r2_ridge"]
    fig, ax = canvas(1124, 640)
    ax.text(40, 50, "How well each predicts who gets wanted later (R², unseen events)", color=CREAM,
            fontsize=21, fontweight="bold", va="center")
    scale = 2400
    bars = [("First 3 speed dates", dates.min(), dates.max(), ORANGE, "4 numbers"),
            ("Full sign-up survey", survey.min(), survey.max(), SAND, "~40 answers")]
    for i, (label, low, high, colour, size) in enumerate(bars):
        y = 140 + i * 190
        ax.text(40, y, label, color=CREAM, fontsize=26, fontweight="bold", va="center")
        ax.text(40, y + 40, size, color=MUTED, fontsize=17, va="center")
        ax.add_patch(FancyBboxPatch((40, y + 70), max(high * scale, 12), 56,
                                    boxstyle="round,pad=0,rounding_size=8", facecolor=colour,
                                    edgecolor="none"))
        text = f"{high:.2f}" if abs(high - low) < 0.01 else f"{low:.2f} to {high:.2f}"
        ax.text(40 + max(high * scale, 12) + 20, y + 98, text, color=colour, fontsize=30,
                fontweight="bold", va="center")
    ax.text(40, 560, "Person 5: adding stated weights leaves prediction unchanged (AUC .83 either way)",
            color=CREAM, fontsize=18, va="center")
    ax.text(40, 605, "Ridge regression, cross-validated by event  ·  women and men shown as a range",
            color=MUTED, fontsize=15, va="center")
    save(fig, "8d_survey")


def revenue_chain():
    """9: how ranking becomes revenue."""
    funnel = table("ext_a5_funnel").set_index("stage")["value"]
    steps = [("Rank for\nboth sides", "+2.5 to +7 pts\nmatches"),
             ("More\nmatches", f"{pct(funnel['Share with no match'])} of people\nget none today"),
             ("Users\nstay", f"No-match users responded\n{pct(funnel['Next-day response, no match'])} "
                              f"vs {pct(funnel['Next-day response, at least one match'])}"),
             ("Users\npay", "Not in our data:\nA/B test the upgrade rate")]
    fig, ax = canvas(1700, 300)
    w, gap = 330, 93
    for i, (head, evidence) in enumerate(steps):
        x = i * (w + gap) + 10
        colour = ORANGE if i == len(steps) - 1 else CREAM
        card(ax, x, 20, w, 140, fill=RUST, edge=colour, lw=2.5 if i == len(steps) - 1 else 1.5)
        ax.text(x + w / 2, 90, head, color=colour, fontsize=26, fontweight="bold", ha="center",
                va="center", linespacing=1.1)
        ax.text(x + w / 2, 225, evidence, color=CREAM, fontsize=17, ha="center", va="center",
                linespacing=1.35)
        if i < len(steps) - 1:
            ax.add_patch(Polygon([(x + w + 18, 75), (x + w + gap - 18, 90), (x + w + 18, 105)],
                                 facecolor=ORANGE, edgecolor="none"))
    save(fig, "9_chain")


def icon(ax, kind, cx, cy, colour):
    if kind == "heart":
        ax.add_patch(Circle((cx - 15, cy - 8), 18, facecolor=colour))
        ax.add_patch(Circle((cx + 15, cy - 8), 18, facecolor=colour))
        ax.add_patch(Polygon([(cx - 32, cy - 2), (cx + 32, cy - 2), (cx, cy + 34)], facecolor=colour))
    elif kind == "stopwatch":
        ax.add_patch(Circle((cx, cy + 6), 32, facecolor="none", edgecolor=colour, linewidth=5))
        ax.add_patch(FancyBboxPatch((cx - 9, cy - 40), 18, 10, boxstyle="round,pad=0,rounding_size=3",
                                    facecolor=colour, edgecolor="none"))
        ax.plot([cx, cx + 14], [cy + 6, cy - 10], color=colour, linewidth=5, solid_capstyle="round")
    elif kind == "star":
        import numpy as np
        angles = np.linspace(-np.pi / 2, 1.5 * np.pi, 11)[:-1]
        radii = [38 if i % 2 == 0 else 16 for i in range(10)]
        ax.add_patch(Polygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in zip(radii, angles)],
                             facecolor=colour))
    elif kind == "megaphone":
        ax.add_patch(Polygon([(cx - 30, cy - 10), (cx + 26, cy - 34), (cx + 26, cy + 30),
                              (cx - 30, cy + 10)], facecolor=colour))
        ax.add_patch(FancyBboxPatch((cx - 40, cy - 10), 14, 20, boxstyle="round,pad=0,rounding_size=3",
                                    facecolor=colour, edgecolor="none"))
        ax.plot([cx - 22, cx - 14], [cy + 10, cy + 32], color=colour, linewidth=6, solid_capstyle="round")


def decision_tiles():
    """9: the four decisions."""
    lift = table("ext_s3_lifts")
    pick = lift[(lift.slots == 1) & (lift.comparison == "optimised minus random")].iloc[0]
    tiles = [("heart", "Rank for matches", "Optimise for both sides;\ncap mass-likers"),
             ("stopwatch", "Shorten sign-up", "Learn from the first\nfew interactions"),
             ("star", "Sell a Daily Pick", f"One coordinated pick:\n+{pick['lift_pp']:.0f} pts vs random"),
             ("megaphone", "Market on choice", "Lead with looks, fun and\nshared interests")]
    fig, ax = canvas(1700, 340)
    w, gap = 390, 39
    for i, (kind, head, body) in enumerate(tiles):
        x = i * (w + gap) + 8
        card(ax, x, 10, w, 320)
        ax.add_patch(Circle((x + 70, 85), 50, facecolor="#5e1f1f", edgecolor=ORANGE, linewidth=2))
        icon(ax, kind, x + 70, 85, ORANGE)
        ax.text(x + 140, 85, f"{i + 1}", color=MUTED, fontsize=40, style="italic", va="center")
        ax.text(x + 30, 185, head, color=CREAM, fontsize=27, fontweight="bold", va="center")
        ax.text(x + 30, 262, body, color=CREAM, fontsize=19, va="center", linespacing=1.35)
    save(fig, "9_tiles")


# ------------------------------------------------------- full-slide layouts
# 1920x1080 backgrounds. The deck's own serif and sans text boxes sit on top in
# Canva, so each layout leaves their slots empty (noted in each docstring).

MAROON, DEEP = "#4c0f12", "#3a0b0e"


def slide(bg=MAROON):
    fig, ax = canvas(1920, 1080)
    fig.patch.set_facecolor(bg)
    ax.add_patch(plt.Rectangle((0, 0), 1920, 1080, facecolor=bg, edgecolor="none", zorder=-10))
    return fig, ax


def save_slide(fig, name):
    fig.savefig(FIGURES / f"deck_full_{name}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def asterisk(ax, cx, cy, r, colour=RUST, lw=5):
    import numpy as np
    for a in np.linspace(0, np.pi, 4, endpoint=False):
        ax.plot([cx - r * np.cos(a), cx + r * np.cos(a)], [cy - r * np.sin(a), cy + r * np.sin(a)],
                color=colour, linewidth=lw, solid_capstyle="round")


def rotate(points, cx, cy, deg):
    import numpy as np
    t = np.deg2rad(deg)
    return [(cx + (x - cx) * np.cos(t) - (y - cy) * np.sin(t),
             cy + (x - cx) * np.sin(t) + (y - cy) * np.cos(t)) for x, y in points]


def slide_8a_ring():
    """Slots: title top-left (40,40); '+3 pts' centred in ring (660-1260, 500);
    headline bottom (60, 985)."""
    row = table("customer_value_top1_agreement").set_index("segment").loc["all"]
    same = row["share_same_top1"]
    fig, ax = slide()
    cx, cy, r = 960, 600, 330
    ax.add_patch(Circle((cx, cy), r + 30, facecolor=DEEP, edgecolor="none"))
    ax.add_patch(Wedge((cx, cy), r, 90, 90 + 360 * same, width=78, facecolor=SAND, edgecolor="none"))
    ax.add_patch(Wedge((cx, cy), r, 90 + 360 * same + 1.2, 450 - 1.2, width=78, facecolor=ORANGE,
                       edgecolor="none"))
    ax.text(cx, cy + 135, "more yeses on the top pick", color=MUTED, fontsize=20, ha="center", va="center")
    # left label (sand arc), right label (orange arc)
    ax.plot([cx - r - 20, 520], [cy - 40, cy - 40], color=SAND, linewidth=2)
    ax.text(500, cy - 75, pct(same), color=SAND, fontsize=56, fontweight="bold", ha="right", va="center")
    ax.text(500, cy, "same #1 either way.\nNo change for them.", color=CREAM, fontsize=21, ha="right",
            va="top", linespacing=1.4)
    ax.plot([cx + r + 20, 1400], [cy + 120, cy + 120], color=ORANGE, linewidth=2)
    ax.text(1420, cy + 85, pct(1 - same), color=ORANGE, fontsize=56, fontweight="bold", va="center")
    ax.text(1420, cy + 160, "#1 changes.\nThe lift lives here.", color=CREAM, fontsize=21, va="top",
            linespacing=1.4)
    asterisk(ax, 1780, 260, 26)
    save_slide(fig, "8a_ring")


def tilted_phone(ax, cx, cy, deg, label, rate, colour, glow):
    w, h = 430, 620
    corners = rotate([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
                      (cx - w / 2, cy + h / 2)], cx, cy, deg)
    if glow:
        for k, a in ((26, 0.12), (12, 0.22)):
            big = rotate([(cx - w / 2 - k, cy - h / 2 - k), (cx + w / 2 + k, cy - h / 2 - k),
                          (cx + w / 2 + k, cy + h / 2 + k), (cx - w / 2 - k, cy + h / 2 + k)], cx, cy, deg)
            ax.add_patch(Polygon(big, facecolor=ORANGE, alpha=a, edgecolor="none", joinstyle="round"))
    ax.add_patch(Polygon(corners, facecolor=DEEP, edgecolor=colour, linewidth=5, joinstyle="round"))
    (nx, ny), = rotate([(cx, cy - h / 2 + 30)], cx, cy, deg)
    notch = rotate([(nx - 50, ny - 7), (nx + 50, ny - 7), (nx + 50, ny + 7), (nx - 50, ny + 7)], nx, ny, deg)
    ax.add_patch(Polygon(notch, facecolor="#ffffff", alpha=0.3, edgecolor="none"))
    (ax_, ay), = rotate([(cx, cy - 130)], cx, cy, deg)
    ax.add_patch(Circle((ax_, ay), 75, facecolor=colour, edgecolor="none"))
    (hx, hy), = rotate([(cx, cy - 152)], cx, cy, deg)
    ax.add_patch(Circle((hx, hy), 28, facecolor=CREAM, edgecolor="none"))
    (sx, sy), = rotate([(cx, cy - 92)], cx, cy, deg)
    ax.add_patch(Wedge((sx, sy), 48, 180 + deg, 360 + deg, facecolor=CREAM, edgecolor="none"))
    for dy, text, size, col, weight in ((-10, label, 22, CREAM, "normal"), (95, pct(rate), 86, colour, "bold"),
                                        (185, "said yes", 24, CREAM, "normal")):
        (tx, ty), = rotate([(cx, cy + dy)], cx, cy, deg)
        ax.text(tx, ty, text, color=col, fontsize=size, fontweight=weight, ha="center", va="center",
                rotation=-deg)


def slide_8b_faceoff():
    """Slots: title top-left (40,40); headline (45,150); '+10 pts' ribbon (560-1360, 905)."""
    row = table("customer_value_top1_agreement").set_index("segment").loc["different_top1"]
    boot = table("customer_value_bootstrap")
    ci = boot[(boot.segment == "different_top1") & (boot.k == 1)].iloc[0]
    fig, ax = slide()
    tilted_phone(ax, 560, 540, -7, "Survey's #1 pick", row["stated_top1_yes_rate"], SAND, False)
    tilted_phone(ax, 1360, 540, 7, "Behaviour's #1 pick", row["revealed_top1_yes_rate"], ORANGE, True)
    ax.add_patch(Circle((960, 530), 78, facecolor=RUST, edgecolor=CREAM, linewidth=3))
    ax.text(960, 530, "VS", color=CREAM, fontsize=40, fontweight="bold", style="italic", ha="center",
            va="center")
    ax.add_patch(Polygon([(500, 900), (1420, 900), (1380, 960), (1420, 1020), (500, 1020), (540, 960)],
                         facecolor=RUST, edgecolor=ORANGE, linewidth=3))
    ax.text(1900, 1060, f"{int(row['n_people'])} people whose #1 differs  ·  95% CI "
            f"+{100 * ci['ci_low']:.0f} to +{100 * ci['ci_high']:.0f} pts", color=MUTED, fontsize=14,
            ha="right", va="center")
    save_slide(fig, "8b_faceoff")


def slide_8c_race():
    """Slots: title top-left (40,40); '34%' right column (1390,250); headline (1395,500)."""
    ranking = table("ext_a5_ranking")
    top3 = ranking[ranking.top_k == 3].set_index("rule")
    direct = table("ext_a5_direct_vs_two_sided")
    direct = direct[(direct.seed == 42) & (direct.top_k == 3)].set_index("metric")
    rates = {rule: (top3.loc[rule, "match_rate"], top3.loc[rule, "yes_rate"]) for rule in top3.index}
    rates["direct"] = (direct.loc["Match rate", "direct_model"], direct.loc["Yes rate", "direct_model"])
    benefit = table("ext_a5_who_benefits").set_index(["tier", "rule"])
    before = benefit.loc[("Least chosen third", "Stated weights"), "no_match_share"]
    after = benefit.loc[("Least chosen third", "Both likely to say yes"), "no_match_share"]
    lanes = [("Random order", "Random order", MUTED), ("Survey weights", "Stated weights", SAND),
             ("Behaviour weights", "Revealed weights", SAND),
             ("Optimise for matches", "direct", blend(ORANGE, MAROON, 0.45)),
             ("Rank for both sides", "Both likely to say yes", ORANGE)]
    fig, ax = slide()
    x0, scale, step, height = 470, 20.0, 124, 104
    ax.text(60, 215, "MATCH RATE IN EACH PERSON'S TOP 3", color=MUTED, fontsize=16, fontweight="bold",
            va="center")
    for i, (label, rule, colour) in enumerate(lanes):
        y = 250 + i * step
        mid = y + height / 2
        ax.add_patch(FancyBboxPatch((40, y), 1300, height, boxstyle="round,pad=0,rounding_size=20",
                                    facecolor=DEEP, edgecolor="none"))
        ax.text(70, mid - 17, label, color=CREAM, fontsize=20, fontweight="bold" if colour == ORANGE else "normal",
                va="center")
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
    # sticker
    cx, cy = 1630, 870
    ax.add_patch(Circle((cx, cy), 140, facecolor=ORANGE, edgecolor=CREAM, linewidth=4))
    for dy, text, size, weight in ((-62, "LEAST-CHOSEN USERS", 13, "bold"), (-30, "with no match", 15, "normal"),
                                   (22, f"{pct(before)} {ARROW} {pct(after)}", 28, "bold")):
        (tx, ty), = rotate([(cx, cy + dy)], cx, cy, -10)
        ax.text(tx, ty, text, color=MAROON, fontsize=size, fontweight=weight, ha="center", va="center",
                rotation=10)
    ax.text(60, 1035, "Pre-date info, unseen events  ·  both sides beats one match model by +1 to +4 pts "
            "across 5 splits  ·  costs 3-5 pts of yeses  ·  cap mass-likers", color=MUTED, fontsize=14,
            va="center")
    save_slide(fig, "8c_race")


def blend(colour, background, amount):
    """Opaque mix, so overlapping shapes don't show seams the way alpha does."""
    import matplotlib.colors as mc
    a, b = mc.to_rgb(colour), mc.to_rgb(background)
    return tuple(amount * x + (1 - amount) * y for x, y in zip(a, b))


def heart(ax, cx, cy, colour, size=1.0, alpha=1.0):
    """Filled heart centred near (cx, cy); size 1 is about 66 px wide."""
    s = size
    if alpha < 1:
        colour, alpha = blend(colour, MAROON, alpha), 1.0
    for dx in (-15, 15):
        ax.add_patch(Circle((cx + dx * s, cy - 8 * s), 18 * s, facecolor=colour, edgecolor="none", alpha=alpha))
    ax.add_patch(Polygon([(cx - 32 * s, cy - 2 * s), (cx + 32 * s, cy - 2 * s), (cx, cy + 34 * s)],
                         facecolor=colour, edgecolor="none", alpha=alpha))


def slide_8c0_wasted():
    """Slots: title top-left (40,40); '61%' right column (1390,250); headline (1395,480);
    pill top-right. Every number from ext_a5_endings."""
    endings = table("ext_a5_endings").set_index("view")
    unreturned = endings.loc["All dates", "unreturned_yes_share"]
    n_faded = round(unreturned * 100)
    before = endings.loc["Top 3: Stated weights", "only_i_said_yes"]
    after = endings.loc["Top 3: Both likely to say yes", "only_i_said_yes"]
    men = endings.loc["All dates: men choosing", "unreturned_yes_share"]
    women = endings.loc["All dates: women choosing", "unreturned_yes_share"]
    fig, ax = slide()
    ax.text(60, 215, "EVERY 100 YESES", color=MUTED, fontsize=16, fontweight="bold", va="center")
    x0, y0, cell = 95, 265, 64
    for i in range(100):
        row, col = divmod(i, 10)
        returned = i < 100 - n_faded
        heart(ax, x0 + col * cell, y0 + row * cell, ORANGE if returned else SAND, size=0.72,
              alpha=1.0 if returned else 0.28)
    legend_x = 790
    heart(ax, legend_x + 30, 330, ORANGE, size=0.9)
    ax.text(legend_x + 85, 315, f"{100 - n_faded} came back", color=CREAM, fontsize=30, fontweight="bold",
            va="center")
    ax.text(legend_x + 85, 360, "both said yes: a match", color=MUTED, fontsize=18, va="center")
    heart(ax, legend_x + 30, 500, SAND, size=0.9, alpha=0.28)
    ax.text(legend_x + 85, 485, f"{n_faded} never returned", color=CREAM, fontsize=30, fontweight="bold",
            va="center")
    ax.text(legend_x + 85, 530, "effort with no payoff", color=MUTED, fontsize=18, va="center")
    # the fix
    card(ax, legend_x - 10, 640, 520, 240, fill=DEEP, edge=ORANGE, lw=2.5, radius=26)
    ax.text(legend_x + 20, 685, "THE FIX: RANK FOR BOTH SIDES", color=ORANGE, fontsize=15, fontweight="bold",
            va="center")
    ax.text(legend_x + 20, 730, "top-3 slots where only you said yes", color=MUTED, fontsize=16, va="center")
    ax.text(legend_x + 20, 800, f"{pct(before)} {ARROW} {pct(after)}", color=CREAM, fontsize=48,
            fontweight="bold", va="center")
    ax.text(legend_x + 20, 852, "survey weights  vs  both sides", color=MUTED, fontsize=15, va="center")
    ax.text(60, 1035, f"All {int(endings.loc['All dates', 'count']):,} dates  ·  men's yeses {pct(men)} unreturned, "
            f"women's {pct(women)}  ·  top 3: pre-date info, unseen events", color=MUTED, fontsize=14,
            va="center")
    save_slide(fig, "8c0_wasted")


def slide_8d_seesaw():
    """Slots: title top-left (40,40); '3 > 40' right (1300,250); headline (1305,480)."""
    import numpy as np
    pred = table("ext_a3_prediction")
    survey = pred[pred.predictors == "Full sign-up survey"]["r2_ridge"]
    dates = pred[pred.predictors == "First 3 speed dates: ratings and yeses"]["r2_ridge"]
    fig, ax = slide()
    fx, fy, half, tilt = 600, 800, 480, -9  # beam pivot; negative = left end down
    t = np.deg2rad(-tilt)

    def beam(d, up=0):  # point along beam at distance d from pivot, 'up' px above it
        return fx + d * np.cos(t) + up * np.sin(t), fy + d * np.sin(t) - up * np.cos(t)

    ax.add_patch(Polygon([(fx, fy + 6), (fx - 80, fy + 170), (fx + 80, fy + 170)], facecolor=RUST,
                         edgecolor=CREAM, linewidth=2))
    ax.add_patch(FancyBboxPatch((fx - 220, fy + 170), 440, 18, boxstyle="round,pad=0,rounding_size=9",
                                facecolor=RUST, edgecolor="none"))
    p1, p2 = beam(-half), beam(half)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=CREAM, linewidth=14, solid_capstyle="round")
    # left: three orange blocks
    for k in range(3):
        d = -half + 70 + k * 105
        corners = [beam(d - 45, 12), beam(d + 45, 12), beam(d + 45, 102), beam(d - 45, 102)]
        ax.add_patch(Polygon(corners, facecolor=ORANGE, edgecolor=MAROON, linewidth=3))
        cx_, cy_ = beam(d, 57)
        ax.text(cx_, cy_, f"{k + 1}", color=MAROON, fontsize=34, fontweight="bold", ha="center",
                va="center", rotation=tilt)
    # right: tall wobbly stack of ~40 survey sheets
    rng = np.random.default_rng(7)
    for k in range(40):
        d = half - 130 + rng.normal(0, 6)
        corners = [beam(d - 85, 12 + k * 11), beam(d + 85, 12 + k * 11), beam(d + 85, 21 + k * 11),
                   beam(d - 85, 21 + k * 11)]
        ax.add_patch(Polygon(corners, facecolor=SAND if k % 2 else "#e8d7b9", edgecolor=MAROON,
                             linewidth=1))
    lx, ly = beam(-half + 175, -80)
    ax.text(lx, ly + 40, "First 3 speed dates", color=CREAM, fontsize=24, fontweight="bold", ha="center",
            va="center")
    ax.text(lx, ly + 85, f"predicts R² {dates.max():.2f}", color=ORANGE, fontsize=30, fontweight="bold",
            ha="center", va="center")
    rx, ry = beam(half - 130, 470)
    ax.text(rx, ry - 70, "~40 sign-up answers", color=CREAM, fontsize=24, fontweight="bold", ha="center",
            va="center")
    ax.text(rx, ry - 25, f"predicts R² {survey.min():.2f}-{survey.max():.2f}", color=SAND, fontsize=30,
            fontweight="bold", ha="center", va="center")
    ax.text(60, 1035, "Predicting who gets wanted later, on unseen events  ·  Person 5: adding stated "
            "weights leaves prediction unchanged (AUC .83 either way)", color=MUTED, fontsize=14, va="center")
    save_slide(fig, "8d_seesaw")


def slide_9_flywheel():
    """Slots: title (60,150) w~1000; A/B line bottom (60,1000)."""
    import numpy as np
    funnel = table("ext_a5_funnel").set_index("stage")["value"]
    exposure = table("ext_a5_exposure").set_index("rule")["top_20pct_share"]
    lift = table("ext_s3_lifts")
    pick = lift[(lift.slots == 1) & (lift.comparison == "optimised minus random")].iloc[0]
    fig, ax = slide()
    cx, cy, R = 1290, 600, 270
    steps = [("Rank for\nmatches", "+2.5 to +7 pts matches", 90),
             ("More\nmatches", f"{pct(funnel['Share with no match'])} get\nnone today", 0),
             ("Users\nstay", f"no-match users: {pct(funnel['Next-day response, no match'])} vs "
                             f"{pct(funnel['Next-day response, at least one match'])} came back", -90),
             ("Users\npay", f"+{100 * funnel['Change in date chance per extra match']:.0f} pts date chance "
                            "per match", 180)]
    ax.add_patch(Circle((cx, cy), R, facecolor="none", edgecolor=RUST, linewidth=22))
    for k in range(4):  # arrowheads clockwise between nodes
        a = np.deg2rad(45 - 90 * k)
        x, y = cx + R * np.cos(a), cy - R * np.sin(a)
        d = np.array([np.sin(a), np.cos(a)])  # clockwise tangent in screen coords
        n = np.array([np.cos(a), -np.sin(a)])
        tip = np.array([x, y]) + d * 30
        ax.add_patch(Polygon([tip, np.array([x, y]) - d * 20 + n * 30, np.array([x, y]) - d * 20 - n * 30],
                             facecolor=ORANGE))
    for label, evidence, deg in steps:
        a = np.deg2rad(deg)
        x, y = cx + R * np.cos(a), cy - R * np.sin(a)
        last = deg == 180
        ax.add_patch(Circle((x, y), 98, facecolor=RUST, edgecolor=ORANGE if last else CREAM, linewidth=4))
        ax.text(x, y, label, color=ORANGE if last else CREAM, fontsize=21, fontweight="bold", ha="center",
                va="center", linespacing=1.1)
        ox, oy = cx + (R + 150) * np.cos(a), cy - (R + 125) * np.sin(a)
        ha = "center" if deg in (90, -90) else ("left" if deg == 0 else "right")
        if deg == 0:
            ox = x + 115
        if deg == 180:
            ox = x - 115
        ax.text(ox, oy if deg in (90, -90) else y, evidence.replace(" per", "\nper").replace(" came", "\ncame"),
                color=CREAM, fontsize=16, ha=ha, va="center", linespacing=1.3)
    ax.text(cx, cy - 18, "THE", color=MUTED, fontsize=16, ha="center", va="center")
    ax.text(cx, cy + 18, "LOOP", color=MUTED, fontsize=26, fontweight="bold", ha="center", va="center")
    decisions = [("heart", "Rank for matches", f"cap mass-likers; top 20% now get "
                                               f"~{100 * exposure['Revealed weights']:.0f}% of slots"),
                 ("stopwatch", "Shorten sign-up", "learn from the first few interactions"),
                 ("star", "Sell a Daily Pick", f"one coordinated pick: +{pick['lift_pp']:.0f} pts vs random"),
                 ("megaphone", "Market on choice", "lead with looks, fun, shared interests")]
    for i, (kind, head, body) in enumerate(decisions):
        y = 400 + i * 135
        ax.add_patch(Circle((110, y), 48, facecolor=DEEP, edgecolor=ORANGE, linewidth=2.5))
        icon(ax, kind, 110, y, ORANGE)
        ax.text(185, y - 20, f"{i + 1}. {head}", color=CREAM, fontsize=24, fontweight="bold", va="center")
        ax.text(185, y + 22, body, color=MUTED, fontsize=17, va="center")
    save_slide(fig, "9_flywheel")


def slide_10_poster():
    """Cream poster. Slots: quote (110,230) w1150; right maroon column 1340-1880;
    'Thank you' (110,900)."""
    fig, ax = slide(CREAM)
    ax.text(70, 330, "“", color=ORANGE, fontsize=420, fontweight="bold", va="center", alpha=0.9,
            family="serif")
    ax.add_patch(FancyBboxPatch((1340, 170), 540, 760, boxstyle="round,pad=0,rounding_size=34",
                                facecolor=MAROON, edgecolor="none"))
    ax.plot([1380, 1840], [520, 520], color=RUST, linewidth=2)
    ax.plot([110, 1200], [860, 860], color=MAROON, linewidth=2)
    asterisk(ax, 1250, 860, 22, colour=ORANGE)
    save_slide(fig, "10_poster")


def arrow(ax, x1, y1, x2, y2, colour=CREAM, lw=3, head=16):
    """Straight arrow in slide pixels."""
    import numpy as np
    ax.plot([x1, x2], [y1, y2], color=colour, linewidth=lw, solid_capstyle="round")
    angle = np.arctan2(y2 - y1, x2 - x1)
    left = (x2 - head * np.cos(angle - 0.45), y2 - head * np.sin(angle - 0.45))
    right = (x2 - head * np.cos(angle + 0.45), y2 - head * np.sin(angle + 0.45))
    ax.add_patch(Polygon([(x2, y2), left, right], facecolor=colour, edgecolor="none"))


def slide_8c_model():
    """How 'rank for both sides' works. Slot: serif title top-left (108,105).
    Weights read from core_model_results; the Alex/Ben example is illustrative."""
    core = table("core_model_results").set_index(["gender", "attribute"])["revealed"]
    looks = f"{core['Women', 'attr']:.0f}\u2013{core['Men', 'attr']:.0f}"
    fun = f"{core['Women', 'fun']:.0f}\u2013{core['Men', 'fun']:.0f}"
    fig, ax = slide()
    rose = blend(RUST, CREAM, 0.85)

    # inputs
    card(ax, 60, 290, 400, 470, fill=DEEP, edge=MUTED, lw=1.5, radius=24)
    ax.text(90, 330, "EACH PROFILE", color=MUTED, fontsize=15, fontweight="bold", va="center")
    ax.text(90, 365, "known before you meet", color=MUTED, fontsize=13, va="center")
    inputs = ["How others rated them", "(looks, fun, sincerity\u2026)", "Their yes rate",
              "Their stated preferences", "Age gap, hobby overlap,", "same race"]
    ys = [430, 465, 530, 595, 660, 695]
    bold = {0, 2, 3, 4}
    for y, text in zip(ys, inputs):
        ax.text(90, y, text, color=CREAM if inputs.index(text) in bold else MUTED, fontsize=17, va="center")

    # model A and B
    for top, colour, tag, question, lines, learns in (
            (270, SAND, "A", "Will YOU say yes?",
             ["Fixed-effects logit on everyone's yeses vs noes, + pair facts",
              f"same method as Jiayi's weights (looks {looks}, fun {fun})"],
             "learns from: did you say yes"),
            (540, rose, "B", "Will THEY say yes?",
             ["Logistic regression on their yes rate,",
              "stated preferences, age gap, hobbies"],
             "learns from: did they say yes to you")):
        card(ax, 580, top, 640, 230, fill=DEEP, edge=colour, lw=2.5, radius=24)
        ax.add_patch(Circle((630, top + 52), 28, facecolor=colour, edgecolor="none"))
        ax.text(630, top + 53, tag, color=MAROON, fontsize=22, fontweight="bold", ha="center", va="center")
        ax.text(678, top + 52, question, color=CREAM, fontsize=24, fontweight="bold", va="center")
        ax.text(610, top + 115, lines[0], color=CREAM, fontsize=15, va="center")
        ax.text(610, top + 148, lines[1], color=CREAM, fontsize=15, va="center")
        ax.text(610, top + 195, learns, color=colour, fontsize=14, fontstyle="italic", va="center")
    arrow(ax, 470, 470, 570, 390)
    arrow(ax, 470, 580, 570, 650)

    # combine
    arrow(ax, 1230, 385, 1320, 480, colour=ORANGE, lw=4)
    arrow(ax, 1230, 655, 1320, 560, colour=ORANGE, lw=4)
    card(ax, 1330, 430, 260, 180, fill=ORANGE, edge=CREAM, lw=3, radius=28)
    ax.text(1460, 490, "P(A) \u00d7 P(B)", color=MAROON, fontsize=26, fontweight="bold", ha="center", va="center")
    ax.text(1460, 548, "chance of a match", color=MAROON, fontsize=16, ha="center", va="center")

    # sort -> top 3
    arrow(ax, 1600, 520, 1660, 520, colour=ORANGE, lw=4)
    ax.text(1630, 488, "sort", color=MUTED, fontsize=13, ha="center", va="center")
    for i in range(3):
        y = 400 + i * 85
        card(ax, 1680, y, 200, 68, fill=DEEP if i else ORANGE, edge=CREAM, lw=1.5, radius=16)
        ax.add_patch(Circle((1718, y + 34), 18, facecolor=CREAM if i else MAROON, edgecolor="none"))
        ax.text(1718, y + 35, str(i + 1), color=MAROON if i else CREAM, fontsize=15, fontweight="bold",
                ha="center", va="center")
        ax.plot([1750, 1855], [y + 27, y + 27], color=CREAM if i else MAROON, linewidth=5, alpha=0.6,
                solid_capstyle="round")
        ax.plot([1750, 1820], [y + 45, y + 45], color=CREAM if i else MAROON, linewidth=5, alpha=0.35,
                solid_capstyle="round")
    ax.text(1780, 375, "TOP 3 SHOWN", color=MUTED, fontsize=14, fontweight="bold", ha="center", va="center")

    # worked example
    card(ax, 60, 820, 1820, 140, fill=DEEP, edge="none", lw=0, radius=22)
    ax.text(95, 858, "EXAMPLE (ILLUSTRATIVE)", color=MUTED, fontsize=13, fontweight="bold", va="center")
    ax.text(95, 912, "Alex", color=CREAM, fontsize=20, fontweight="bold", va="center")
    ax.text(185, 912, f"0.90 \u00d7 0.20 = 0.18   {ARROW} #2", color=CREAM, fontsize=20, va="center")
    ax.text(640, 912, "Ben", color=ORANGE, fontsize=20, fontweight="bold", va="center")
    ax.text(720, 912, f"0.70 \u00d7 0.60 = 0.42   {ARROW} #1", color=ORANGE, fontsize=20, fontweight="bold",
            va="center")
    ax.text(1180, 912, "You like Alex more, but Ben likes you back.", color=MUTED, fontsize=17,
            fontstyle="italic", va="center")
    ax.text(60, 1035, "Pre-date info only  \u00b7  trained on 20 events, tested on the 21st (\u00d721)  \u00b7  "
            "men and women modelled separately  \u00b7  judged on real matches in each top 3",
            color=MUTED, fontsize=14, va="center")
    save_slide(fig, "8c_model")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for build in (ranking_donut, twist_phones, match_bars, survey_vs_dates, revenue_chain,
                  decision_tiles, slide_8a_ring, slide_8b_faceoff, slide_8c0_wasted, slide_8c_model, slide_8c_race, slide_8d_seesaw,
                  slide_9_flywheel, slide_10_poster):
        build()
    print("Wrote outputs/figures/deck_*.png")


if __name__ == "__main__":
    main()
