"""Extended analyses about surveys, beliefs and the robustness of Part 3.

Each public function takes a Context and returns a dict with `tables` (name to
DataFrame, saved as outputs/tables/ext_<name>.csv) and `rows` (entries for the
summary menu). IDs match docs/extended_analysis_report.md.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from statsmodels.stats.anova import anova_lm

from .constants import ATTR_LABELS, ATTRS, SEED, STATED, TABLES_DIR
from .core_model import conditional_objective
from .ext_features import (CONSENSUS, GENDERS, add_consensus, boot_mean, bootstrap_fits,
                           demean, fit_clogit, importance, importance_draws, make_strata,
                           normalised_weights, p_from_draws, person_loglik, relative_weights,
                           summarise_draws)
from .io import project_root

ATTRACTIVE = ATTRS.index("attr")
LABELS = dict(zip(ATTRS, ATTR_LABELS))
SELF_RATED = ["attr", "sinc", "intel", "fun", "amb"]
QUESTIONS = {
    "1_1": "What I look for (sign-up)",
    "4_1": "What others of my gender look for",
    "2_1": "What the opposite sex looks for",
    "1_2": "What I look for (day after)",
    "1_3": "What I look for (3-4 weeks later)",
    "7_2": "What drove my decisions (day after)",
    "7_3": "What drove my decisions (3-4 weeks later)",
}


@dataclass
class Context:
    """Data and shared bootstrap draws for one run."""
    data: pd.DataFrame
    sample: pd.DataFrame
    n_boot: int
    cache: dict = field(default_factory=dict)

    def core_fit(self, code):
        """Part 3's model with bootstrap draws, computed once per gender."""
        if code not in self.cache:
            strata = make_strata(self.sample[self.sample["gender"] == code], ATTRS)
            self.cache[code] = (strata, *bootstrap_fits(strata, self.n_boot, SEED + code))
        return self.cache[code]


def finding(analysis, answer, estimate=None, ci=(None, None), p=None, unit="", sample="",
            table="", primary=False):
    return dict(id=analysis, answer=answer, estimate=estimate, ci_low=ci[0], ci_high=ci[1],
                p=p, unit=unit, sample=sample, table=table, primary=primary)


def boot_gap(first, second, n_boot, seed):
    """Difference in means between two independent groups, with interval."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(first, float), np.asarray(second, float)
    draws = (a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
             - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1))
    return dict(estimate=float(a.mean() - b.mean()), **summarise_draws(draws))


def ols_cluster(formula, frame, groups):
    return smf.ols(formula, frame).fit(cov_type="cluster", cov_kwds={"groups": groups})


# --------------------------------------------------------- A1 survey trust


def a1_survey_trust(ctx):
    """Is the gap lasting, how do traits trade off, and do genders differ?"""
    published = pd.read_csv(project_root() / TABLES_DIR / "core_model_results.csv")
    gaps = published[["gender", "label", "stated", "revealed", "gap_pp", "gap_ci_low",
                      "gap_ci_high"]].copy()

    people = ctx.data.drop_duplicates("iid").set_index("iid")
    people = people[~people["wave"].between(6, 9)]
    before = normalised_weights(people, "1_1")
    retest_rows = []
    for suffix, label in [("1_2", "Day after"), ("1_3", "3-4 weeks later")]:
        later = normalised_weights(people, suffix)
        both = before.add_prefix("a_").join(later.add_prefix("b_")).dropna()
        rng = np.random.default_rng(SEED)
        for attribute in ATTRS:
            a, b = both[f"a_{attribute}"].to_numpy(), both[f"b_{attribute}"].to_numpy()
            idx = rng.integers(0, len(a), (ctx.n_boot, len(a)))
            draws = [np.corrcoef(a[i], b[i])[0, 1] for i in idx]
            retest_rows.append(dict(
                interval=label, attribute=LABELS[attribute], people=len(a),
                correlation=np.corrcoef(a, b)[0, 1], ci_low=np.quantile(draws, 0.025),
                ci_high=np.quantile(draws, 0.975), mean_shift_pp=(b - a).mean()))
    retest = pd.DataFrame(retest_rows)

    trade_rows, gender_rows = [], []
    for code, name in GENDERS:
        _, base, _, raw = ctx.core_fit(code)
        for other in ("intel", "fun", "shar"):
            j = ATTRS.index(other)
            draws = raw[:, ATTRACTIVE] / raw[:, j]
            trade_rows.append(dict(
                gender=name, one_point_of="Attractive", equals_points_of=LABELS[other],
                ratio=base.beta_raw[ATTRACTIVE] / base.beta_raw[j],
                ci_low=np.quantile(draws, 0.025), ci_high=np.quantile(draws, 0.975),
                share_of_draws_with_positive_denominator=(raw[:, j] > 0).mean()))
    _, women, women_std, _ = ctx.core_fit(0)
    _, men, men_std, _ = ctx.core_fit(1)
    for j, attribute in enumerate(ATTRS):
        draws = men_std[:, j] - women_std[:, j]
        gender_rows.append(dict(attribute=LABELS[attribute], men_coef=men.beta[j],
                                women_coef=women.beta[j],
                                men_minus_women=men.beta[j] - women.beta[j],
                                **summarise_draws(draws)))
    trade, gender = pd.DataFrame(trade_rows), pd.DataFrame(gender_rows)

    attr_r = retest[(retest["attribute"] == "Attractive")
                    & (retest["interval"] == "3-4 weeks later")].iloc[0]
    women_gap = gaps[(gaps["gender"] == "Women") & (gaps["label"] == "Attractive")].iloc[0]
    men_gap = gaps[(gaps["gender"] == "Men") & (gaps["label"] == "Attractive")].iloc[0]
    women_t = trade[(trade["gender"] == "Women") & (trade["equals_points_of"] == "Intelligent")].iloc[0]
    men_t = trade[(trade["gender"] == "Men") & (trade["equals_points_of"] == "Intelligent")].iloc[0]
    attr_g = gender[gender["attribute"] == "Attractive"].iloc[0]
    sinc_g = gender[gender["attribute"] == "Sincere"].iloc[0]
    intel_g = gender[gender["attribute"] == "Intelligent"].iloc[0]
    rows = [
        finding("A1a", f"Attractiveness counts {women_gap.gap_pp:.1f} (women) and "
                       f"{men_gap.gap_pp:.1f} (men) points more in choices than in the survey",
                women_gap.gap_pp, (women_gap.gap_ci_low, women_gap.gap_ci_high), None,
                "importance points, women", "452 people", "core_model_results.csv"),
        finding("A1b", f"Stated attractiveness weight is stable after 3-4 weeks "
                       f"(r={attr_r.correlation:.2f}), so the gap is lasting bias, not noise",
                attr_r.correlation, (attr_r.ci_low, attr_r.ci_high), None, "correlation",
                f"{attr_r.people} people, 100-point waves", "a1_retest"),
        finding("A1c", f"One attractiveness point trades for {women_t.ratio:.1f} intelligence "
                       f"points for women and {men_t.ratio:.1f} for men",
                men_t.ratio, (men_t.ci_low, men_t.ci_high), None, "points, men", "452 people",
                "a1_tradeoffs"),
        finding("A1d", f"Gender differences are not clear: men minus women is "
                       f"{attr_g.men_minus_women:+.2f} on attractiveness (p={attr_g.p:.2f}), "
                       f"{sinc_g.men_minus_women:+.2f} on sincerity (p={sinc_g.p:.2f}) and "
                       f"{intel_g.men_minus_women:+.2f} on intelligence (p={intel_g.p:.2f})",
                attr_g.men_minus_women, (attr_g.ci_low, attr_g.ci_high), attr_g.p,
                "standardised coefficient", "452 people", "a1_gender_test", primary=True),
    ]
    return dict(tables=dict(a1_gaps=gaps, a1_retest=retest, a1_tradeoffs=trade,
                            a1_gender_test=gender), rows=rows)


# ------------------------------------------------------- A2 better survey


def question_bootstrap(ctx, frame, questions, seed, complete=False, baseline="1_1"):
    """Compare survey questions with revealed weights on the same people.

    Revealed weights are refitted in every draw, so question-to-question
    differences carry the model's uncertainty. Each question is compared with
    `baseline` on the people who answered both; with complete=True every
    question uses only people who answered all of them.
    `2_1` asks about the opposite sex, so its target is that group's choices.
    """
    strata, fits, weights = {}, {}, {}
    for code, _ in GENDERS:
        side = frame[frame["gender"] == code]
        strata[code] = make_strata(side, ATTRS)
        fits[code] = fit_clogit(strata[code])
        people = side.drop_duplicates("iid").set_index("iid").reindex(strata[code].ids)
        weights[code] = {q: normalised_weights(people, q).to_numpy() for q in questions}
        if complete:
            usable = np.all([~np.isnan(w).any(axis=1) for w in weights[code].values()], axis=0)
            weights[code] = {q: np.where(usable[:, None], w, np.nan)
                             for q, w in weights[code].items()}

    def target(code, question, revealed):
        return revealed[1 - code] if question == "2_1" else revealed[code]

    def stats(revealed, multiplicity):
        out = {}
        for code, _ in GENDERS:
            for question in questions:
                w = weights[code][question]
                base = weights[code][baseline]
                have = ~np.isnan(w).any(axis=1)
                both = have & ~np.isnan(base).any(axis=1)
                m = multiplicity[code]
                goal = target(code, question, revealed)
                mean_all = (m[have, None] * w[have]).sum(axis=0) / m[have].sum()
                mean_q = (m[both, None] * w[both]).sum(axis=0) / m[both].sum()
                mean_own = (m[both, None] * base[both]).sum(axis=0) / m[both].sum()
                own_goal = revealed[code]
                out[code, question] = np.r_[
                    goal - mean_all,
                    np.abs(goal - mean_all).mean(),
                    np.abs(goal - mean_q).mean() - np.abs(own_goal - mean_own).mean(),
                    (goal - mean_q)[ATTRACTIVE] - (own_goal - mean_own)[ATTRACTIVE]]
        return out

    n_people = {c: len(strata[c].yes_counts) for c, _ in GENDERS}
    point_rev = {c: importance(fits[c].beta) for c, _ in GENDERS}
    point = stats(point_rev, {c: np.ones(n_people[c]) for c, _ in GENDERS})
    draws = {k: np.zeros((ctx.n_boot, 9)) for k in point}
    rng = np.random.default_rng(seed)
    for b in range(ctx.n_boot):
        multiplicity, revealed = {}, {}
        for code, _ in GENDERS:
            multiplicity[code] = np.bincount(rng.integers(0, n_people[code], n_people[code]),
                                             minlength=n_people[code]).astype(float)
            revealed[code] = importance(fit_clogit(strata[code], multiplicity[code],
                                                   fits[code].beta).beta)
        for key, value in stats(revealed, multiplicity).items():
            draws[key][b] = value

    metric_rows, attribute_rows = [], []
    for (code, question), value in point.items():
        name = dict(GENDERS)[code]
        answered = (~np.isnan(weights[code][question]).any(axis=1)).sum()
        goal = target(code, question, point_rev)
        gap = summarise_draws(draws[code, question][:, ATTRACTIVE])
        mad = summarise_draws(draws[code, question][:, 6])
        metric_rows.append(dict(
            gender=name, question=QUESTIONS[question], suffix=question, people=answered,
            attractive_share=goal[ATTRACTIVE] - value[ATTRACTIVE],
            attractive_target=goal[ATTRACTIVE], attractive_gap=value[ATTRACTIVE],
            gap_ci_low=gap["ci_low"], gap_ci_high=gap["ci_high"], mean_abs_error=value[6],
            error_ci_low=mad["ci_low"], error_ci_high=mad["ci_high"]))
        if question != baseline:
            shift = summarise_draws(draws[code, question][:, 7])
            attr_shift = summarise_draws(draws[code, question][:, 8])
            metric_rows[-1].update(
                error_change_vs_baseline=value[7], error_change_ci_low=shift["ci_low"],
                error_change_ci_high=shift["ci_high"], error_change_p=shift["p"],
                gap_change_vs_baseline=value[8], gap_change_ci_low=attr_shift["ci_low"],
                gap_change_ci_high=attr_shift["ci_high"], gap_change_p=attr_shift["p"])
        for j, attribute in enumerate(ATTRS):
            interval = summarise_draws(draws[code, question][:, j])
            attribute_rows.append(dict(
                gender=name, question=QUESTIONS[question], suffix=question,
                attribute=LABELS[attribute], answer=goal[j] - value[j], actual=goal[j],
                gap=value[j], gap_ci_low=interval["ci_low"], gap_ci_high=interval["ci_high"]))
    return pd.DataFrame(metric_rows), pd.DataFrame(attribute_rows)


def a2_better_survey(ctx):
    """Which question lands closest to behaviour, and which carries individual signal?"""
    recent = ctx.sample[ctx.sample["wave"] >= 10]
    wording, attributes = question_bootstrap(
        ctx, recent, ["1_1", "4_1", "2_1", "1_2", "1_3"], SEED + 10)
    ctx.cache["wording_attributes"] = attributes
    ladder, _ = question_bootstrap(ctx, ctx.sample[ctx.sample["wave"] >= 15],
                                   ["1_1", "1_2", "7_2"], SEED + 11, complete=True, baseline="1_2")

    # Individual signal: does a person's answer predict their own slope?
    rows = []
    frame = recent.copy()
    z_columns = [f"z_{a}" for a in ATTRS]
    for attribute, column in zip(ATTRS, z_columns):
        frame[column] = demean(frame, [attribute])[attribute] / frame[attribute].std()
    frame["dd"] = demean(frame, ["dec"])["dec"]
    people = frame.drop_duplicates("iid").set_index("iid")
    for suffix in ("1_1", "4_1"):
        answer = normalised_weights(people, suffix)
        use = frame[frame["iid"].isin(answer.dropna().index)].copy()
        tilt_columns = []
        for attribute, column in zip(ATTRS, z_columns):
            value = use["iid"].map(answer[attribute])
            tilt = f"t_{attribute}"
            use[tilt] = use[column] * (value - value.mean()) / value.std()
            tilt_columns.append(tilt)
        fit = ols_cluster("dd ~ -1 + " + " + ".join(z_columns + tilt_columns), use, use["iid"])
        for attribute, column, tilt in zip(ATTRS, z_columns, tilt_columns):
            rows.append(dict(question=QUESTIONS[suffix], attribute=LABELS[attribute],
                             average_slope=fit.params[column], tilt_per_sd=fit.params[tilt],
                             t_stat=fit.tvalues[tilt], p=fit.pvalues[tilt],
                             people=use["iid"].nunique()))
    signal = pd.DataFrame(rows)

    # Waves 6-9 used a 1-10 rating; the rest allocated 100 points.
    people = ctx.sample.drop_duplicates("iid").set_index("iid")
    rating = people["wave"].between(6, 9)
    format_rows = []
    for code, name in GENDERS:
        side = people["gender"] == code
        result = boot_gap(people.loc[side & rating, "attr1_1"],
                          people.loc[side & ~rating, "attr1_1"], ctx.n_boot, SEED + code)
        format_rows.append(dict(
            gender=name, rating_scale=people.loc[side & rating, "attr1_1"].mean(),
            hundred_points=people.loc[side & ~rating, "attr1_1"].mean(),
            people_rating=int((side & rating).sum()), people_points=int((side & ~rating).sum()),
            **{f"difference_{k}" if k == "estimate" else k: v for k, v in result.items()}))
    survey_format = pd.DataFrame(format_rows)

    def row(table, gender, suffix):
        return table[(table["gender"] == gender) & (table["suffix"] == suffix)].iloc[0]

    w_own, w_peer = row(wording, "Women", "1_1"), row(wording, "Women", "4_1")
    m_own, m_peer = row(wording, "Men", "1_1"), row(wording, "Men", "4_1")
    w_back, w_after = row(ladder, "Women", "7_2"), row(ladder, "Women", "1_2")
    m_back, m_after = row(ladder, "Men", "7_2"), row(ladder, "Men", "1_2")
    own = signal[(signal["question"] == QUESTIONS["1_1"]) & (signal["attribute"] == "Attractive")].iloc[0]
    peer = signal[(signal["question"] == QUESTIONS["4_1"]) & (signal["attribute"] == "Attractive")].iloc[0]
    men_format = survey_format[survey_format["gender"] == "Men"].iloc[0]
    n_recent = recent["iid"].nunique()
    rows = [
        finding("A2a", f"Asking what others want shrinks the attractiveness gap from "
                       f"{w_own.attractive_gap:.1f} to {w_peer.attractive_gap:.1f} (women) and "
                       f"{m_own.attractive_gap:.1f} to {m_peer.attractive_gap:.1f} (men)",
                w_peer.error_change_vs_baseline,
                (w_peer.error_change_ci_low, w_peer.error_change_ci_high), w_peer.error_change_p,
                "error change, women", f"{n_recent} people, waves 10-21", "a2_wording",
                primary=True),
        finding("A2b", "The same result for men",
                m_peer.error_change_vs_baseline,
                (m_peer.error_change_ci_low, m_peer.error_change_ci_high), m_peer.error_change_p,
                "error change, men", f"{n_recent} people, waves 10-21", "a2_wording",
                primary=True),
        finding("A2c", f"On the same people, 'what drove my decisions' gives attractiveness "
                       f"{w_back.attractive_share:.1f} (women) and {m_back.attractive_share:.1f} "
                       f"(men), against {w_after.attractive_share:.1f} and "
                       f"{m_after.attractive_share:.1f} for 'what I look for'",
                w_back.gap_change_vs_baseline,
                (w_back.gap_change_ci_low, w_back.gap_change_ci_high), w_back.gap_change_p,
                "gap change vs 'what I look for', women", f"{w_back.people} women, waves 15-21",
                "a2_ladder", primary=True),
        finding("A2c2", "The same comparison for men", m_back.gap_change_vs_baseline,
                (m_back.gap_change_ci_low, m_back.gap_change_ci_high), m_back.gap_change_p,
                "gap change vs 'what I look for', men", f"{m_back.people} men, waves 15-21",
                "a2_ladder", primary=True),
        finding("A2d", f"Only the direct question separates individuals: t={own.t_stat:.1f} vs "
                       f"t={peer.t_stat:.1f} on attractiveness",
                own.tilt_per_sd, (None, None), own.p, "slope change per SD", f"{own.people} people",
                "a2_individual_signal", primary=True),
        finding("A2e", f"A 1-10 rating scale lowers men's stated attractiveness weight by "
                       f"{-men_format.difference_estimate:.1f} points against 100-point allocation",
                men_format.difference_estimate, (men_format.ci_low, men_format.ci_high),
                men_format.p, "points, men", "Waves not randomised", "a2_survey_format"),
    ]
    return dict(tables=dict(a2_wording=wording, a2_wording_attributes=attributes,
                            a2_ladder=ladder, a2_individual_signal=signal,
                            a2_survey_format=survey_format), rows=rows)


# ------------------------------------------------- A4 market understanding


def a4_market_understanding(ctx):
    """Do people know what others want, how they are seen, and who likes them?"""
    beliefs = ctx.cache["wording_attributes"]
    beliefs = beliefs[beliefs["suffix"] == "2_1"].drop(columns=["question", "suffix"]).copy()
    beliefs = beliefs.rename(columns={"gender": "believer", "answer": "believed",
                                      "actual": "choosers_actual"})
    beliefs["choosers"] = beliefs["believer"].map({"Women": "Men", "Men": "Women"})

    rated = ctx.data.dropna(subset=["prob", "dec_o", "dec"])
    bins = rated.groupby(pd.cut(rated["prob"], [0, 2, 4, 6, 8, 10]), observed=True).agg(
        rows=("dec_o", "size"), partner_said_yes=("dec_o", "mean"),
        i_said_yes=("dec", "mean")).reset_index()
    bins["prob"] = bins["prob"].astype(str)

    by_person = [g.index.to_numpy() for _, g in rated.groupby("iid")]
    prob, partner, own = (rated[c].to_numpy(float) for c in ("prob", "dec_o", "dec"))
    positions = pd.Series(np.arange(len(rated)), index=rated.index)
    by_person = [positions.loc[ix].to_numpy() for ix in by_person]
    rng = np.random.default_rng(SEED)

    def scores(rows):
        p, y = prob[rows] / 10, partner[rows]
        base = y.mean()
        return (roc_auc_score(y, prob[rows]), roc_auc_score(own[rows], prob[rows]),
                np.mean((p - y)**2), np.mean((base - y)**2))

    draws = []
    for _ in range(ctx.n_boot):
        chosen = rng.integers(0, len(by_person), len(by_person))
        draws.append(scores(np.concatenate([by_person[i] for i in chosen])))
    draws = np.array(draws)
    point = scores(np.arange(len(rated)))
    summary_rows = [
        dict(measure="AUC: guess vs partner's real answer", value=point[0],
             **summarise_draws(draws[:, 0] - 0.5)),
        dict(measure="AUC: guess vs own decision", value=point[1],
             **summarise_draws(draws[:, 1] - 0.5)),
        dict(measure="AUC gap: own decision minus partner's answer", value=point[1] - point[0],
             **summarise_draws(draws[:, 1] - draws[:, 0])),
        dict(measure="Brier score of guesses", value=point[2],
             **summarise_draws(draws[:, 2])),
        dict(measure="Brier score of predicting the average", value=point[3],
             **summarise_draws(draws[:, 3])),
        dict(measure="Brier gap: guesses minus average (positive is worse)",
             value=point[2] - point[3], **summarise_draws(draws[:, 2] - draws[:, 3])),
    ]
    calibration = pd.DataFrame(summary_rows)

    optimism_rows = []
    for code, name in GENDERS:
        side = rated[rated["gender"] == code]
        person = side.groupby("iid").apply(
            lambda g: (g["prob"] / 10 - g["dec_o"]).mean(), include_groups=False)
        result = boot_mean(person, ctx.n_boot, SEED + code)
        optimism_rows.append(dict(gender=name, mean_guess=(side["prob"] / 10).mean(),
                                  actual_yes=side["dec_o"].mean(), people=len(person),
                                  overestimate=result["estimate"], ci_low=result["ci_low"],
                                  ci_high=result["ci_high"], p=result["p"]))
    optimism = pd.DataFrame(optimism_rows)

    people = ctx.data.drop_duplicates("iid").set_index("iid")
    received = ctx.data.groupby("iid").agg(
        **{f"{a}_received": (f"{a}_o", "mean") for a in SELF_RATED},
        raters=("attr_o", "count"), match_rate=("match", "mean"), own_yes=("dec", "mean"),
        yes_received=("dec_o", "mean"), matches=("match", "sum"))
    joined = people.join(received)
    joined = joined[joined["raters"] >= 5]
    image_rows = []
    for suffix, label in [("5_1", "How I think others see me"), ("3_1", "How I rate myself")]:
        for attribute in SELF_RATED:
            pair = joined[[f"{attribute}{suffix}", f"{attribute}_received"]].dropna()
            mine, theirs = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, len(mine), (ctx.n_boot, len(mine)))
            corr = [np.corrcoef(mine[i], theirs[i])[0, 1] for i in idx]
            image_rows.append(dict(question=label, attribute=LABELS[attribute], people=len(pair),
                                   believed=mine.mean(), received=theirs.mean(),
                                   share_overrating=(mine > theirs).mean(),
                                   correlation=np.corrcoef(mine, theirs)[0, 1],
                                   corr_ci_low=np.quantile(corr, 0.025),
                                   corr_ci_high=np.quantile(corr, 0.975)))
    image = pd.DataFrame(image_rows)

    use = joined.dropna(subset=["attr5_1", "attr_received"]).copy()
    effect_rows = []
    for outcome, label in [("own_yes", "Own yes rate"), ("match_rate", "Match rate"),
                           ("match_es", "Matches expected"), ("satis_2", "Satisfaction")]:
        sub = use.dropna(subset=[outcome])
        fit = smf.ols(f"{outcome} ~ attr5_1 + attr_received + C(gender)", sub).fit(cov_type="HC1")
        effect_rows.append(dict(outcome=label, people=len(sub),
                                believed_attractiveness=fit.params["attr5_1"],
                                p=fit.pvalues["attr5_1"],
                                actual_attractiveness=fit.params["attr_received"],
                                actual_p=fit.pvalues["attr_received"]))
    effects = pd.DataFrame(effect_rows)

    expected = people.join(received).dropna(subset=["match_es"])
    difference = boot_mean(expected["match_es"] - expected["matches"], ctx.n_boot, SEED)
    interest = beliefs[beliefs["attribute"] == "Shared interests"]
    women, men = optimism.iloc[0], optimism.iloc[1]
    self_attr = image[(image["question"] == "How I think others see me")]
    brier = calibration.iloc[5]
    auc_gap = calibration.iloc[2]
    expected_row = effects[effects["outcome"] == "Matches expected"].iloc[0]
    match_row = effects[effects["outcome"] == "Match rate"].iloc[0]
    worst = interest.sort_values("gap", ascending=False).iloc[0]
    rows = [
        finding("A4a", f"Guesses about a partner's yes score worse than predicting the average "
                       f"(Brier {point[2]:.3f} vs {point[3]:.3f})",
                brier.value, (brier.ci_low, brier.ci_high), brier.p, "Brier gap",
                f"{rated['iid'].nunique()} people", "a4_calibration", primary=True),
        finding("A4b", f"Guesses track people's own decision more than the partner's "
                       f"(AUC {point[1]:.2f} vs {point[0]:.2f})",
                auc_gap.value, (auc_gap.ci_low, auc_gap.ci_high), auc_gap.p, "AUC gap",
                f"{rated['iid'].nunique()} people", "a4_calibration", primary=True),
        finding("A4c", f"Men overestimate acceptance by {men.overestimate * 100:.0f} pp, "
                       f"women by {women.overestimate * 100:.0f} pp",
                men.overestimate * 100, (men.ci_low * 100, men.ci_high * 100), men.p, "pp, men",
                f"{int(men.people)} men", "a4_optimism", primary=True),
        finding("A4d", f"Both genders underrate shared interests: the largest miss is "
                       f"{worst.gap:.1f} points ({worst.choosers.lower()}'s choices give it "
                       f"{worst.choosers_actual:.1f}; {worst.believer.lower()} guess "
                       f"{worst.believed:.1f})",
                worst.gap, (worst.gap_ci_low, worst.gap_ci_high), None, "points",
                "Waves 10-21", "a4_beliefs"),
        finding("A4e", f"{self_attr['share_overrating'].min():.0%}-"
                       f"{self_attr['share_overrating'].max():.0%} believe others rate them above "
                       f"what they receive; belief tracks reality weakly "
                       f"(r={self_attr['correlation'].min():.2f} to "
                       f"{self_attr['correlation'].max():.2f})",
                self_attr["correlation"].max(), (None, None), None, "highest correlation",
                f"{int(self_attr['people'].max())} people, waves 10-21", "a4_self_image"),
        finding("A4f", f"Believing you are more attractive raises matches you expect "
                       f"({expected_row.believed_attractiveness:+.2f} per point) but not "
                       f"matches you get ({match_row.believed_attractiveness:+.3f} on match rate)",
                expected_row.believed_attractiveness, (None, None), expected_row.p,
                "matches expected per point", f"{int(expected_row.people)} people",
                "a4_belief_effects", primary=True),
        finding("A4g", f"People expect {expected['match_es'].mean():.2f} matches and get "
                       f"{expected['matches'].mean():.2f}",
                difference["estimate"], (difference["ci_low"], difference["ci_high"]),
                difference["p"], "matches", f"{len(expected)} people", "a4_belief_effects"),
    ]
    return dict(tables=dict(a4_beliefs=beliefs, a4_calibration=calibration,
                            a4_calibration_bins=bins, a4_optimism=optimism,
                            a4_self_image=image, a4_belief_effects=effects), rows=rows)


# ------------------------------------------------------- S1 perceptual map


def stated_segments(sample, k=3):
    """K-means benefit segments on stated weights, one row per person."""
    people = sample.drop_duplicates("iid").set_index("iid")
    weights = people[STATED].to_numpy(dtype=float)
    z = (weights - weights.mean(axis=0)) / weights.std(axis=0)
    labels = KMeans(k, n_init=20, random_state=SEED).fit_predict(z)
    return pd.Series(labels, index=people.index, name="segment"), z


def s1_perceptual_map(ctx):
    """Place people by the ratings they received, then overlay preferences."""
    received = ctx.data.groupby("iid").agg(
        **{a: (f"{a}_o", "mean") for a in ATTRS}, raters=("attr_o", "count"),
        yes_received=("dec_o", "mean"), match_rate=("match", "mean"),
        gender=("gender", "first"))
    received = received[received["raters"] >= 5].dropna()
    z = (received[ATTRS] - received[ATTRS].mean()) / received[ATTRS].std()
    pca = PCA(3, random_state=SEED).fit(z)
    # Fix the sign so axis 1 is overall appeal and axis 2 points to brains.
    signs = np.array([np.sign(pca.components_[0].sum()),
                      np.sign(pca.components_[1, ATTRS.index("intel")]), 1.0])
    loadings = pca.components_ * signs[:, None]
    coordinates = z.to_numpy() @ loadings.T
    received["axis_appeal"], received["axis_brains"] = coordinates[:, 0], coordinates[:, 1]

    loading_table = pd.DataFrame(loadings[:2], columns=[LABELS[a] for a in ATTRS])
    loading_table.insert(0, "axis", ["Overall appeal", "Brains vs looks"])
    loading_table.insert(1, "variance_share", pca.explained_variance_ratio_[:2])

    arrows = []
    people = ctx.sample.drop_duplicates("iid")
    for code, name in GENDERS:
        _, base, std, _ = ctx.core_fit(code)
        stated = people.loc[people["gender"] == code, STATED].mean().to_numpy()
        revealed = importance(base.beta)
        for source, vector in [("Stated", stated), ("Revealed", revealed)]:
            position = loadings[:2] @ (vector - 100 / 6)
            arrows.append(dict(gender=name, source=source, appeal=position[0],
                               brains=position[1]))
    arrows = pd.DataFrame(arrows)

    fit_rows = []
    for code, name in GENDERS:
        side = received[received["gender"] == code]
        fit_rows.append(dict(
            people_rated=name, people=len(side),
            r2_attractive_only=smf.ols("yes_received ~ attr", side).fit().rsquared,
            r2_all_six=smf.ols("yes_received ~ " + " + ".join(ATTRS), side).fit().rsquared,
            r2_two_axes=smf.ols("yes_received ~ axis_appeal + axis_brains", side).fit().rsquared))
    fits = pd.DataFrame(fit_rows)

    segments, _ = stated_segments(ctx.sample)
    joined = received.join(segments, how="inner")
    eta = {axis: smf.ols(f"{axis} ~ C(segment)", joined).fit().rsquared
           for axis in ("axis_appeal", "axis_brains")}
    fits["segment_share_of_appeal_axis"] = eta["axis_appeal"]
    fits["segment_share_of_brains_axis"] = eta["axis_brains"]

    corr = received[["axis_appeal", "axis_brains", "yes_received"]].corr().loc["yes_received"]
    women_arrow = arrows[arrows["gender"] == "Women"].set_index("source")
    rows = [
        finding("S1", f"Two axes: overall appeal ({pca.explained_variance_ratio_[0]:.0%}) and "
                      f"brains vs looks ({pca.explained_variance_ratio_[1]:.0%}). Women's stated "
                      f"arrow sits at {women_arrow.loc['Stated', 'brains']:+.1f} on brains vs "
                      f"looks, their revealed arrow at {women_arrow.loc['Revealed', 'brains']:+.1f}",
                corr["axis_brains"], (None, None), None, "corr, brains axis vs yeses received",
                f"{len(received)} people with 5+ ratings", "s1_axes"),
    ]
    return dict(tables=dict(s1_axes=loading_table, s1_arrows=arrows, s1_fit=fits,
                            s1_people=received.reset_index()[
                                ["iid", "gender", "axis_appeal", "axis_brains", "yes_received",
                                 "match_rate"]]),
                rows=rows)


# ---------------------------------------------------------------- robustness


def r1_consensus(ctx):
    """Halo check: replace a person's own rating with everyone else's."""
    frame = add_consensus(ctx.data)
    keys = ctx.sample[["iid", "pid"]].drop_duplicates()
    frame = frame.merge(keys, on=["iid", "pid"]).copy()
    table = []
    for code, name in GENDERS:
        side = frame[frame["gender"] == code]
        strata = make_strata(side, CONSENSUS)
        base, std, _ = bootstrap_fits(strata, ctx.n_boot, SEED + 20 + code)
        points, draws = importance(base.beta), importance_draws(std)
        own = importance(ctx.core_fit(code)[1].beta)
        for j, attribute in enumerate(ATTRS):
            own_col, other_col = side[attribute], side[CONSENSUS[j]]
            table.append(dict(
                gender=name, attribute=LABELS[attribute], revealed_own_ratings=own[j],
                revealed_consensus=points[j], ci_low=np.quantile(draws[:, j], 0.025),
                ci_high=np.quantile(draws[:, j], 0.975), consensus_coefficient=base.beta[j],
                agreement_own_vs_consensus=own_col.corr(other_col), people=len(strata.ids)))
    table = pd.DataFrame(table)
    attr = table[table["attribute"] == "Attractive"].set_index("gender")
    rows = [finding("R1", f"Using other raters' scores, attractiveness rises to "
                          f"{attr.loc['Women', 'revealed_consensus']:.1f} (women) and "
                          f"{attr.loc['Men', 'revealed_consensus']:.1f} (men), so the headline "
                          f"is not a halo artefact",
                    attr.loc["Women", "revealed_consensus"],
                    (attr.loc["Women", "ci_low"], attr.loc["Women", "ci_high"]), None,
                    "importance, women", f"{int(attr['people'].sum())} people", "r1_consensus")]
    return dict(tables=dict(r1_consensus=table), rows=rows)


def r2_relative_weights(ctx):
    """Importance with no clipping: Johnson's relative weights, within person."""
    table = []
    for code, name in GENDERS:
        side = ctx.sample[ctx.sample["gender"] == code].sort_values("iid", kind="stable")
        x, y = demean(side, ATTRS).to_numpy(), demean(side, ["dec"])["dec"].to_numpy()
        groups = list(side.groupby("iid").indices.values())
        weights = relative_weights(x, y)
        rng = np.random.default_rng(SEED + code)
        draws = np.zeros((ctx.n_boot, len(ATTRS)))
        for b in range(ctx.n_boot):
            rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            raw = relative_weights(x[rows], y[rows])
            draws[b] = 100 * raw / raw.sum()
        share = 100 * weights / weights.sum()
        clipped = importance(ctx.core_fit(code)[1].beta)
        for j, attribute in enumerate(ATTRS):
            table.append(dict(gender=name, attribute=LABELS[attribute], relative_weight=share[j],
                              ci_low=np.quantile(draws[:, j], 0.025),
                              ci_high=np.quantile(draws[:, j], 0.975),
                              clipped_importance=clipped[j], r_squared=weights.sum()))
    table = pd.DataFrame(table)
    attr = table[table["attribute"] == "Attractive"].set_index("gender")
    rows = [finding("R2", f"With no clipping, attractiveness is "
                          f"{attr.loc['Women', 'relative_weight']:.1f} (women) and "
                          f"{attr.loc['Men', 'relative_weight']:.1f} (men), close to the published "
                          f"{attr.loc['Women', 'clipped_importance']:.1f} and "
                          f"{attr.loc['Men', 'clipped_importance']:.1f}",
                    attr.loc["Men", "relative_weight"],
                    (attr.loc["Men", "ci_low"], attr.loc["Men", "ci_high"]), None,
                    "share of explained variance, men", "452 people", "r2_relative_weights")]
    return dict(tables=dict(r2_relative_weights=table), rows=rows)


def r3_dropout_weights(ctx):
    """Reweight retained people to look like everyone who signed up."""
    people = ctx.data.groupby("iid").agg(
        yes_received=("dec_o", "mean"), dates=("dec", "size"), wave=("wave", "first"),
        gender=("gender", "first"), age=("age", "first"), go_out=("go_out", "first"),
        date=("date", "first"))
    people["kept"] = people.index.isin(ctx.sample["iid"]).astype(int)
    people["rating_format"] = people["wave"].between(6, 9).astype(int)
    people = people.dropna(subset=["age", "go_out", "date", "yes_received"])
    model = smf.logit("kept ~ C(gender) + age + go_out + date + yes_received + rating_format"
                      " + dates", people).fit(disp=0)
    people["p_kept"] = model.predict(people)
    table = []
    for code, name in GENDERS:
        side = ctx.sample[(ctx.sample["gender"] == code) & ctx.sample["iid"].isin(people.index)]
        strata = make_strata(side, ATTRS)
        weights = (1 / people.loc[strata.ids, "p_kept"]).clip(upper=10).to_numpy()
        weights = weights / weights.mean()
        plain = importance(fit_clogit(strata).beta)
        weighted = importance(fit_clogit(strata, weights).beta)
        for j, attribute in enumerate(ATTRS):
            table.append(dict(gender=name, attribute=LABELS[attribute], unweighted=plain[j],
                              reweighted=weighted[j], change=weighted[j] - plain[j],
                              largest_weight=weights.max()))
    table = pd.DataFrame(table)
    worst = table["change"].abs().max()
    rows = [finding("R3", f"Reweighting to all sign-ups moves no importance by more than "
                          f"{worst:.1f} points", worst, unit="largest change",
                    sample="Retained people, weighted", table="r3_dropout_weights")]
    return dict(tables=dict(r3_dropout_weights=table), rows=rows)


def r4_excluding_rating_waves(ctx):
    """Part 3's gaps using only waves that used the 100-point question."""
    frame = ctx.sample[~ctx.sample["wave"].between(6, 9)]
    published = pd.read_csv(project_root() / TABLES_DIR / "core_model_results.csv")
    table = []
    for code, name in GENDERS:
        side = frame[frame["gender"] == code]
        strata = make_strata(side, ATTRS)
        base, std, _ = bootstrap_fits(strata, ctx.n_boot, SEED + 30 + code)
        people = side.drop_duplicates("iid").set_index("iid").reindex(strata.ids)
        stated = people[STATED].to_numpy()
        rng = np.random.default_rng(SEED + 40 + code)
        idx = rng.integers(0, len(stated), (ctx.n_boot, len(stated)))
        gaps = importance_draws(std) - stated[idx].mean(axis=1)
        point = importance(base.beta) - stated.mean(axis=0)
        reference = published[published["gender"] == name].set_index("attribute")
        for j, attribute in enumerate(ATTRS):
            table.append(dict(gender=name, attribute=LABELS[attribute], people=len(stated),
                              stated=stated[:, j].mean(), revealed=importance(base.beta)[j],
                              gap=point[j], ci_low=np.quantile(gaps[:, j], 0.025),
                              ci_high=np.quantile(gaps[:, j], 0.975),
                              published_gap=reference.loc[attribute, "gap_pp"]))
    table = pd.DataFrame(table)
    attr = table[table["attribute"] == "Attractive"].set_index("gender")
    rows = [finding("R4", f"On 100-point waves only, the attractiveness gap is "
                          f"{attr.loc['Women', 'gap']:.1f} (women) and {attr.loc['Men', 'gap']:.1f} "
                          f"(men), against the published {attr.loc['Women', 'published_gap']:.1f} "
                          f"and {attr.loc['Men', 'published_gap']:.1f}",
                    attr.loc["Women", "gap"], (attr.loc["Women", "ci_low"], attr.loc["Women", "ci_high"]),
                    None, "points, women", f"{int(attr['people'].sum())} people",
                    "r4_excluding_rating_waves")]
    return dict(tables=dict(r4_excluding_rating_waves=table), rows=rows)


def _m_step(x, strata, weights, start):
    return minimize(conditional_objective, start, jac=True, method="BFGS",
                    args=(x, strata.y, strata.offsets, strata.yes_counts, weights),
                    options={"maxiter": 60, "gtol": 1e-4}).x


def latent_class(strata, x, classes, seed, iterations=60):
    """EM for a mixture of conditional logits. Returns loglik, betas, posteriors."""
    rng = np.random.default_rng(seed)
    n_people = len(strata.yes_counts)
    posterior = rng.dirichlet(np.ones(classes) * 0.5, n_people) if classes > 1 else np.ones((n_people, 1))
    betas = np.zeros((classes, x.shape[1]))
    previous = -np.inf
    total = previous
    for _ in range(iterations):
        for k in range(classes):
            betas[k] = _m_step(x, strata, posterior[:, k] + 1e-9, betas[k])
        shares = posterior.mean(axis=0)
        logs = np.stack([person_loglik(betas[k], x, strata.y, strata.offsets, strata.yes_counts)
                         for k in range(classes)], axis=1) + np.log(shares + 1e-12)
        each = logsumexp(logs, axis=1)
        posterior = np.exp(logs - each[:, None])
        total = float(each.sum())
        if abs(total - previous) < 1e-3:
            break
        previous = total
    return total, betas, posterior


def best_latent_class(strata, x, classes, starts=5):
    results = [latent_class(strata, x, classes, SEED + s) for s in range(1 if classes == 1 else starts)]
    return max(results, key=lambda r: r[0])


def r5_latent_class(ctx):
    """Are there behavioural segments that hold up on unseen dates?"""
    fit_rows, class_rows, holdout_rows = [], [], []
    for code, name in GENDERS:
        side = ctx.sample[ctx.sample["gender"] == code]
        strata = make_strata(side, ATTRS)
        scale = fit_clogit(strata).sd
        x = np.ascontiguousarray(strata.centred / scale)
        people = len(strata.ids)
        stated = side.drop_duplicates("iid").set_index("iid").loc[strata.ids, "attr1_1"].to_numpy()
        for classes in (1, 2, 3, 4):
            total, betas, posterior = best_latent_class(strata, x, classes)
            parameters = 6 * classes + classes - 1
            fit_rows.append(dict(gender=name, classes=classes, loglik=total,
                                 bic=-2 * total + parameters * np.log(people),
                                 aic=-2 * total + 2 * parameters,
                                 smallest_class=posterior.mean(axis=0).min()))
            if classes in (2, 3):
                assigned = posterior.argmax(axis=1)
                for k in range(classes):
                    class_rows.append(dict(
                        gender=name, classes=classes, label=k, people=int((assigned == k).sum()),
                        stated_attractive=stated[assigned == k].mean() if (assigned == k).any() else np.nan,
                        **{LABELS[a]: v for a, v in zip(ATTRS, importance(betas[k]))}))
        for split in range(3):
            rng = np.random.default_rng(SEED + split)
            rows = side.assign(half=rng.random(len(side)) < 0.5)
            varied = rows.groupby(["iid", "half"])["dec"].nunique().unstack()
            rows = rows[rows["iid"].isin(varied[(varied[True] == 2) & (varied[False] == 2)].index)]
            train, test = make_strata(rows[rows["half"]], ATTRS), make_strata(rows[~rows["half"]], ATTRS)
            x_train = np.ascontiguousarray(train.centred / scale)
            x_test = np.ascontiguousarray(test.centred / scale)
            baseline = None
            for classes in (1, 2, 3):
                _, betas, posterior = best_latent_class(train, x_train, classes, starts=4)
                logs = np.stack([person_loglik(betas[k], x_test, test.y, test.offsets, test.yes_counts)
                                 for k in range(classes)], axis=1)
                held_out = logsumexp(logs + np.log(posterior + 1e-12), axis=1).sum()
                baseline = held_out if classes == 1 else baseline
                holdout_rows.append(dict(gender=name, split=split, classes=classes,
                                         people=len(test.ids), held_out_loglik=held_out,
                                         gain_vs_one_class=held_out - baseline))
    fits, holdout = pd.DataFrame(fit_rows), pd.DataFrame(holdout_rows)
    classes_table = pd.DataFrame(class_rows)
    mean_gain = holdout[holdout["classes"] > 1].groupby(["gender", "classes"])["gain_vs_one_class"].mean()
    best = fits.loc[fits.groupby("gender")["bic"].idxmin()].set_index("gender")["classes"]
    rows = [finding("R5", f"Extra behavioural classes do not hold up: held-out fit changes by "
                          f"{mean_gain.min():.1f} to {mean_gain.max():.1f} log-likelihood points "
                          f"(BIC picks {int(best['Women'])} class for women, "
                          f"{int(best['Men'])} for men)",
                    float(mean_gain.max()), unit="best held-out gain vs one class",
                    sample="452 people; 3 random date splits", table="r5_holdout")]
    return dict(tables=dict(r5_fit=fits, r5_classes=classes_table, r5_holdout=holdout), rows=rows)


def r6_stated_segments(ctx):
    """Do segments built from survey answers behave differently?"""
    people = ctx.sample.drop_duplicates("iid").set_index("iid")
    weights = people[STATED].to_numpy(dtype=float)
    z = (weights - weights.mean(axis=0)) / weights.std(axis=0)
    quality = []
    for k in range(2, 7):
        labels = KMeans(k, n_init=20, random_state=SEED).fit_predict(z)
        rng = np.random.default_rng(SEED)
        stability = []
        for _ in range(30):
            rows = rng.integers(0, len(z), len(z))
            refit = KMeans(k, n_init=5, random_state=1).fit(z[rows])
            stability.append(adjusted_rand_score(labels, refit.predict(z)))
        quality.append(dict(segments=k, silhouette=silhouette_score(z, labels),
                            stability_ari=np.mean(stability),
                            smallest_share=np.bincount(labels).min() / len(labels)))
    quality = pd.DataFrame(quality)

    segments, _ = stated_segments(ctx.sample)
    frame = ctx.sample.assign(segment=ctx.sample["iid"].map(segments))
    outcomes = ctx.data.groupby("iid").agg(
        yes_rate=("dec", "mean"), yes_received=("dec_o", "mean"), match_rate=("match", "mean"),
        satisfaction=("satis_2", "first")).join(people[["gender"]], how="inner").join(segments)
    anova_rows = []
    for outcome in ("yes_rate", "yes_received", "match_rate", "satisfaction"):
        sub = outcome if outcome else None
        use = outcomes.dropna(subset=[sub])
        reduced = smf.ols(f"{sub} ~ C(gender)", use).fit()
        full = smf.ols(f"{sub} ~ C(gender) + C(segment)", use).fit()
        anova_rows.append(dict(outcome=sub, p=anova_lm(reduced, full)["Pr(>F)"].iloc[1],
                               added_r2=full.rsquared - reduced.rsquared))
    anova = pd.DataFrame(anova_rows)

    weight_rows, test_rows = [], []
    for code, name in GENDERS:
        side = frame[frame["gender"] == code]
        pooled = fit_clogit(make_strata(side, ATTRS)).loglike
        separate = 0.0
        groups = side["segment"].nunique()
        for segment, group in side.groupby("segment"):
            strata = make_strata(group, ATTRS)
            fit = fit_clogit(strata)
            separate += fit.loglike
            weight_rows.append(dict(
                gender=name, segment=int(segment), people=len(strata.ids),
                stated_attractive=group.drop_duplicates("iid")["attr1_1"].mean(),
                **{LABELS[a]: v for a, v in zip(ATTRS, importance(fit.beta))}))
        statistic = 2 * (separate - pooled)
        test_rows.append(dict(gender=name, lr_statistic=statistic, df=6 * (groups - 1),
                              p=chi2.sf(statistic, 6 * (groups - 1))))
    tests = pd.DataFrame(test_rows)
    three = quality[quality["segments"] == 3].iloc[0]
    women, men = tests.iloc[0], tests.iloc[1]
    match = anova[anova["outcome"] == "match_rate"].iloc[0]
    rows = [
        finding("R6a", f"Survey segments are weak (silhouette {three.silhouette:.2f}) and do not "
                       f"choose differently (p={women.p:.2f} women, p={men.p:.2f} men)",
                women.p, unit="p, women", p=women.p, sample="452 people, 3 segments",
                table="r6_choice_test", primary=True),
        finding("R6b", "The same test for men", men.p, unit="p, men", p=men.p,
                sample="452 people, 3 segments", table="r6_choice_test", primary=True),
        finding("R6c", f"Segments explain {match.added_r2:.1%} of match rate", match.added_r2,
                p=match.p, unit="added R²", sample="452 people", table="r6_outcomes"),
    ]
    return dict(tables=dict(r6_quality=quality, r6_outcomes=anova, r6_revealed_by_segment=pd.DataFrame(weight_rows),
                            r6_choice_test=tests), rows=rows)


# ------------------------------------------------- tested, not recommended


def n1_goals(ctx):
    """Does the stated reason for attending change choices?"""
    groups = {1: "Casual", 2: "Casual", 3: "Serious", 4: "Serious", 5: "Other", 6: "Other"}
    frame = ctx.sample.assign(goal_group=ctx.sample["goal"].map(groups)).dropna(subset=["goal_group"])
    weights, tests = [], []
    for code, name in GENDERS:
        side = frame[frame["gender"] == code]
        pooled = fit_clogit(make_strata(side, ATTRS)).loglike
        separate = 0.0
        for group, rows in side.groupby("goal_group"):
            strata = make_strata(rows, ATTRS)
            fit = fit_clogit(strata)
            separate += fit.loglike
            weights.append(dict(gender=name, goal=group, people=len(strata.ids),
                                yes_rate=rows["dec"].mean(), match_rate=rows["match"].mean(),
                                **{LABELS[a]: v for a, v in zip(ATTRS, importance(fit.beta))}))
        statistic = 2 * (separate - pooled)
        tests.append(dict(gender=name, lr_statistic=statistic, df=12, p=chi2.sf(statistic, 12)))
    weights, tests = pd.DataFrame(weights), pd.DataFrame(tests)
    serious = weights[weights["goal"] == "Serious"].set_index("gender")["people"]
    rows = [finding("N1a", f"Goals: too few serious attendees ({serious['Women']} women, "
                           f"{serious['Men']} men) to support segments (women p={tests.p[0]:.2f})",
                    tests.p[0], p=tests.p[0], unit="p, women", sample="452 people",
                    table="n1_goal_tests", primary=True),
            finding("N1b", f"Men's goals differ at p={tests.p[1]:.3f}, from small groups",
                    tests.p[1], p=tests.p[1], unit="p, men", sample="452 people",
                    table="n1_goal_tests", primary=True)]
    return dict(tables=dict(n1_goal_weights=weights, n1_goal_tests=tests), rows=rows)


def n2_interest_similarity(ctx):
    """Does hobby similarity matter more to those who say interests matter?"""
    frame = ctx.sample.dropna(subset=["int_corr"]).copy()
    frame["similar"] = (frame["int_corr"] - frame["int_corr"].mean()) / frame["int_corr"].std()
    frame["says"] = (frame["shar1_1"] - frame["shar1_1"].mean()) / frame["shar1_1"].std()
    frame["both"] = frame["similar"] * frame["says"]
    table = []
    for outcome in ("dec", "match"):
        for controls, extra in [("No ratings", []), ("With ratings", ATTRS)]:
            columns = [outcome, "similar", "both", *extra]
            within = demean(frame, columns).assign(iid=frame["iid"])
            formula = f"{outcome} ~ -1 + similar + both" + "".join(f" + {a}" for a in extra)
            fit = ols_cluster(formula, within, within["iid"])
            table.append(dict(outcome=outcome, model=controls,
                              similarity_pp_per_sd=fit.params["similar"] * 100,
                              similarity_p=fit.pvalues["similar"],
                              interaction_pp=fit.params["both"] * 100,
                              interaction_p=fit.pvalues["both"], people=frame["iid"].nunique()))
    table = pd.DataFrame(table)
    plain = table[(table["outcome"] == "dec") & (table["model"] == "No ratings")].iloc[0]
    rated = table[(table["outcome"] == "dec") & (table["model"] == "With ratings")].iloc[0]
    rows = [finding("N2", f"Hobby similarity adds {plain.similarity_pp_per_sd:+.1f} pp per SD, "
                          f"{rated.similarity_pp_per_sd:+.1f} pp once ratings are included; it "
                          f"does not matter more to those who say it does",
                    plain.interaction_pp, p=plain.interaction_p, unit="pp interaction",
                    sample="452 people", table="n2_interest_similarity", primary=True)]
    return dict(tables=dict(n2_interest_similarity=table), rows=rows)


def n3_event_size(ctx):
    """Do bigger events shift weight towards easily judged traits?"""
    frame = ctx.sample.copy()
    frame["large"] = (frame["round"] >= 15).astype(int)
    weights = []
    for code, name in GENDERS:
        for large, rows in frame[frame["gender"] == code].groupby("large"):
            strata = make_strata(rows, ATTRS)
            weights.append(dict(gender=name, event="15+ partners" if large else "Under 15 partners",
                                people=len(strata.ids), events=rows["wave"].nunique(),
                                yes_rate=rows["dec"].mean(),
                                **{LABELS[a]: v for a, v in zip(ATTRS, importance(fit_clogit(strata).beta))}))
    z_columns = [f"z_{a}" for a in ATTRS]
    for attribute, column in zip(ATTRS, z_columns):
        frame[column] = (frame[attribute] - frame[attribute].mean()) / frame[attribute].std()
        frame[f"x_{attribute}"] = frame[column] * frame["large"]
    extra = [f"x_{a}" for a in ATTRS]
    within = demean(frame, ["dec", *z_columns, *extra]).assign(wave=frame["wave"])
    fit = ols_cluster("dec ~ -1 + " + " + ".join(z_columns + extra), within, within["wave"])
    tests = pd.DataFrame(dict(attribute=[LABELS[a] for a in ATTRS],
                              small_event_slope=[fit.params[c] for c in z_columns],
                              change_in_large_event=[fit.params[c] for c in extra],
                              p=[fit.pvalues[c] for c in extra]))
    attr = tests.iloc[ATTRACTIVE]
    rows = [finding("N3", f"Event size does not shift weight to looks ({attr.change_in_large_event:+.3f} "
                          f"slope change, p={attr.p:.2f}); only 21 events",
                    attr.change_in_large_event, p=attr.p, unit="slope change",
                    sample="452 people, clustered by event", table="n3_size_tests", primary=True)]
    return dict(tables=dict(n3_size_weights=pd.DataFrame(weights), n3_size_tests=tests), rows=rows)


def n4_polarising(ctx):
    """Given the same mean, do divisive looks earn more yeses?"""
    people = ctx.data.groupby("iid").agg(
        mean_rating=("attr_o", "mean"), spread=("attr_o", "std"), raters=("attr_o", "count"),
        yes_received=("dec_o", "mean"), gender=("gender", "first"),
        **{a: (f"{a}_o", "mean") for a in ATTRS[1:]}).dropna()
    people = people[people["raters"] >= 8]
    table = []
    for code, name in GENDERS:
        side = people[people["gender"] == code].copy()
        for column in ("mean_rating", "spread"):
            side[column] = (side[column] - side[column].mean()) / side[column].std()
        fit = smf.ols("yes_received ~ mean_rating + I(mean_rating**2) + spread + "
                      + " + ".join(ATTRS[1:]), side).fit(cov_type="HC1")
        table.append(dict(people_rated=name, people=len(side), spread_effect=fit.params["spread"],
                          p=fit.pvalues["spread"]))
    table = pd.DataFrame(table)
    rows = [finding("N4", f"Spread in looks has opposite signs by gender "
                          f"({table.spread_effect[0]:+.3f} and {table.spread_effect[1]:+.3f})",
                    table.spread_effect[0], p=table.p[0], unit="yes rate per SD, women rated",
                    sample=f"{int(table.people.sum())} people", table="n4_polarising", primary=True)]
    return dict(tables=dict(n4_polarising=table), rows=rows)


def n5_instrumental_variables(ctx):
    """Instrument a person's rating with other people's ratings of that partner."""
    frame = add_consensus(ctx.data).merge(ctx.sample[["iid", "pid"]], on=["iid", "pid"])
    frame = frame.dropna(subset=[*ATTRS, *CONSENSUS])

    def two_stage(x, z, y):
        projected = z @ np.linalg.solve(z.T @ z, z.T @ x)
        return np.linalg.solve(projected.T @ x, projected.T @ y)

    table = []
    for code, name in GENDERS:
        side = frame[frame["gender"] == code].sort_values("iid", kind="stable").reset_index(drop=True)
        x = demean(side, ATTRS).to_numpy() / side[ATTRS].std().to_numpy()
        z = demean(side, CONSENSUS).to_numpy() / side[CONSENSUS].std().to_numpy()
        y = demean(side, ["dec"])["dec"].to_numpy()
        point = two_stage(x, z, y)
        first_stage = []
        for j in range(len(ATTRS)):
            others = np.delete(z, j, 1)
            resid_z = z[:, j] - others @ np.linalg.lstsq(others, z[:, j], rcond=None)[0]
            resid_x = x[:, j] - others @ np.linalg.lstsq(others, x[:, j], rcond=None)[0]
            slope = resid_z @ resid_x / (resid_z @ resid_z)
            error = resid_x - slope * resid_z
            first_stage.append(slope**2 * (resid_z @ resid_z) / (error @ error / len(error)))
        groups = list(side.groupby("iid").indices.values())
        rng = np.random.default_rng(SEED + code)
        draws = []
        for _ in range(ctx.n_boot):
            rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            draws.append(two_stage(x[rows], z[rows], y[rows]))
        draws = importance_draws(np.array(draws))
        shares = importance(point)
        for j, attribute in enumerate(ATTRS):
            table.append(dict(gender=name, attribute=LABELS[attribute], iv_importance=shares[j],
                              ci_low=np.quantile(draws[:, j], 0.025),
                              ci_high=np.quantile(draws[:, j], 0.975),
                              first_stage_f=first_stage[j]))
    table = pd.DataFrame(table)
    attr = table[table["attribute"] == "Attractive"]
    rows = [finding("N5", f"Instrumental variables are too noisy to use: attractiveness interval "
                          f"{attr.ci_low.min():.0f} to {attr.ci_high.max():.0f}; the weakest "
                          f"first stage has F={table.first_stage_f.min():.0f}",
                    float(attr.iv_importance.iloc[0]), (attr.ci_low.iloc[0], attr.ci_high.iloc[0]),
                    unit="importance, women", sample="452 people", table="n5_instrumental_variables")]
    return dict(tables=dict(n5_instrumental_variables=table), rows=rows)


def n8_self_insight(ctx):
    """For how many people does their own survey beat population weights?"""
    frame = ctx.sample.copy()
    frame["survey_score"] = (frame[STATED].to_numpy() * frame[ATTRS].to_numpy()).sum(axis=1)
    frame["population_score"] = 0.0
    for code, _ in GENDERS:
        side = frame["gender"] == code
        frame.loc[side, "population_score"] = ctx.core_fit(code)[1].score(frame[side], ATTRS)
    scores = frame.groupby("iid").apply(lambda g: pd.Series(dict(
        survey_auc=roc_auc_score(g["dec"], g["survey_score"]),
        population_auc=roc_auc_score(g["dec"], g["population_score"]))), include_groups=False)
    people = ctx.data.drop_duplicates("iid").set_index("iid").join(scores, how="inner")
    people = people.join(ctx.data.groupby("iid").agg(
        match_rate=("match", "mean"), yes_received=("dec_o", "mean"), own_yes=("dec", "mean")))
    summary = pd.DataFrame([dict(
        people=len(scores), survey_auc=scores["survey_auc"].mean(),
        population_auc=scores["population_auc"].mean(),
        survey_better=(scores["survey_auc"] > scores["population_auc"]).mean(),
        tied=(scores["survey_auc"] == scores["population_auc"]).mean())])
    traits = people.dropna(subset=["age", "go_out", "date", "exphappy", "imprace", "imprelig"])
    who = smf.ols("survey_auc ~ C(gender) + age + go_out + date + yes_received + imprelig"
                  " + imprace + exphappy", traits).fit()
    match = smf.ols("match_rate ~ survey_auc + C(gender) + yes_received + own_yes",
                    people).fit(cov_type="HC1")
    summary["traits_r2"] = who.rsquared
    summary["match_link"] = match.params["survey_auc"]
    summary["match_link_p"] = match.pvalues["survey_auc"]
    row = summary.iloc[0]
    rows = [finding("N8", f"A person's survey beats population weights for {row.survey_better:.0%} "
                          f"of people ({row.tied:.0%} tie); traits do not identify them "
                          f"(R²={row.traits_r2:.2f})",
                    row.survey_better, unit="share of people", sample="452 people, post-date ratings",
                    table="n8_self_insight")]
    return dict(tables=dict(n8_self_insight=summary), rows=rows)


def n9_stated_race_importance(ctx):
    """Sensitive item: does stated importance predict same-race choices?"""
    frame = ctx.sample.dropna(subset=["imprace", "samerace"]).copy()
    frame["centred"] = frame["imprace"] - frame["imprace"].mean()
    frame["same_x_stated"] = frame["samerace"] * frame["centred"]
    table = []
    for code, name in GENDERS:
        side = frame[frame["gender"] == code]
        columns = ["dec", "samerace", "same_x_stated", *ATTRS]
        within = demean(side, columns).assign(iid=side["iid"])
        fit = ols_cluster("dec ~ -1 + samerace + same_x_stated + " + " + ".join(ATTRS), within,
                          within["iid"])
        table.append(dict(gender=name, people=side["iid"].nunique(),
                          mean_stated=side.drop_duplicates("iid")["imprace"].mean(),
                          same_race_pp=fit.params["samerace"] * 100,
                          same_race_p=fit.pvalues["samerace"],
                          per_stated_point_pp=fit.params["same_x_stated"] * 100,
                          per_stated_point_p=fit.pvalues["same_x_stated"]))
    table = pd.DataFrame(table)
    rows = [finding("N9a", f"Stated importance of same-race dating does predict behaviour: "
                           f"{table.per_stated_point_pp[0]:+.1f} pp per point (women), "
                           f"{table.per_stated_point_pp[1]:+.1f} pp (men). Sensitive; frame with care",
                    table.per_stated_point_pp[0], p=table.per_stated_point_p[0], unit="pp, women",
                    sample="452 people", table="n9_stated_race_importance", primary=True),
            finding("N9b", "The same test for men", table.per_stated_point_pp[1],
                    p=table.per_stated_point_p[1], unit="pp, men", sample="452 people",
                    table="n9_stated_race_importance", primary=True)]
    return dict(tables=dict(n9_stated_race_importance=table), rows=rows)


ANALYSES = [a1_survey_trust, a2_better_survey, a4_market_understanding, s1_perceptual_map,
            r1_consensus, r2_relative_weights, r3_dropout_weights, r4_excluding_rating_waves,
            r5_latent_class, r6_stated_segments, n1_goals, n2_interest_similarity, n3_event_size,
            n4_polarising, n5_instrumental_variables, n8_self_insight, n9_stated_race_importance]
