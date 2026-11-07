"""Extended analyses about ranking, markets and what can be predicted.

Each public function takes a Context and returns `tables` and `rows`, as in
src/ext_survey.py. Read the leakage note in src/ext_features.py before changing
any ranking feature: two early versions of this file produced results that
looked strong or absurd purely because of feature construction.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from scipy.optimize import linear_sum_assignment, minimize
from scipy.sparse.linalg import lsqr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import ATTRS, SEED, STATED
from .ext_features import (CONSENSUS, GENDERS, HISTORY, PAIR, PARTNER_STATED, SPLIT_FOLD,
                           add_consensus, add_pair_features, add_split_fold, boot_mean,
                           core_sample, demean, fit_clogit, make_strata, split_history,
                           summarise_draws)
from .ext_survey import finding, ols_cluster

N_PERMUTATIONS = 1000
SEEDS = (SEED, 7, 123, 2024, 99)
HOBBIES = ("sports tvsports exercise dining museums art hiking gaming clubbing reading tv "
           "theater movies concerts music shopping yoga").split()
RULE_LABELS = {"random": "Random order", "stated": "Stated weights",
               "revealed": "Revealed weights", "revealed_pair": "Revealed + pair facts",
               "popularity": "Popularity", "partner_only": "Partner likely to say yes",
               "reciprocal": "Both likely to say yes"}


def top_k(frame, rule, k):
    """Per-person mean outcomes on the top k rows of a rule."""
    ranked = frame.sort_values(["iid", rule, "tiebreak"], ascending=[True, False, True])
    return ranked.groupby("iid").head(k).groupby("iid")[["dec", "match", "dec_o"]].mean()


def paired_lift(first, second, column, n_boot, seed=SEED):
    """Mean person-level difference, in percentage points."""
    result = boot_mean((first[column] - second[column]).to_numpy() * 100, n_boot, seed)
    return result


def partner_features(frame, mode):
    """What predicts the partner's yes, from facts known before the meeting.

    Across one person's list that person is fixed, so what varies is the
    partner: how readily they say yes, plus fit with what they stated.
    """
    x = pd.DataFrame({"int": frame["int_corr"], "age": frame["agegap"], "race": frame["samerace"]})
    for attribute in ATTRS:
        x[attribute] = frame[PARTNER_STATED[attribute]].fillna(16.7) / 100
    if mode == "selectivity_only":
        x = x[[]].copy()
    if mode != "no_selectivity":
        x["j_sel"] = frame["j_sel"]
    return x.fillna(0)


def ranking_frame(ctx, seed=SEED, mode="full"):
    """Score each person's list on events the models never saw."""
    frame = add_pair_features(add_split_fold(ctx.data, seed))
    sample = core_sample(frame).dropna(subset=SPLIT_FOLD).copy()
    sample["tiebreak"] = np.random.default_rng(seed).random(len(sample))
    sample["random"] = sample["tiebreak"]
    parts = []
    for wave in sorted(sample["wave"].unique()):
        for code, _ in GENDERS:
            train = sample[(sample["wave"] != wave) & (sample["gender"] == code)]
            test = sample[(sample["wave"] == wave) & (sample["gender"] == code)].copy()
            if test.empty:
                continue
            test["stated"] = (test[STATED].to_numpy() * test[SPLIT_FOLD].to_numpy()).sum(axis=1)
            test["post_stated"] = (test[STATED].to_numpy() * test[ATTRS].to_numpy()).sum(axis=1)
            test["post_revealed"] = fit_clogit(make_strata(train, ATTRS)).score(test, ATTRS)
            test["revealed"] = fit_clogit(make_strata(train, SPLIT_FOLD)).score(test, SPLIT_FOLD)
            columns = SPLIT_FOLD + PAIR
            test["revealed_pair"] = fit_clogit(make_strata(train, columns)).score(test, columns)
            test["popularity"] = test["sf_dec"]
            model = LogisticRegression(C=1.0, max_iter=1000)
            model.fit(partner_features(train, mode), train["dec_o"])
            test["partner_only"] = model.predict_proba(partner_features(test, mode))[:, 1]
            # Own side: rescale utility to this person's yes rate; constant
            # within a list, so it cannot steer the ranking between partners.
            rate = test.groupby("iid")["dec"].transform("mean").clip(0.02, 0.98)
            centred = test["revealed_pair"] - test.groupby("iid")["revealed_pair"].transform("mean")
            own = 1 / (1 + np.exp(-(np.log(rate / (1 - rate)) + centred)))
            test["own_side"] = own
            test["reciprocal"] = np.log(own) + np.log(test["partner_only"])
            parts.append(test)
    return pd.concat(parts, ignore_index=True)


def gini(values):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    return float((2 * np.arange(1, n + 1) - n - 1) @ values / (n * values.sum()))


def exposure_table(scored, rules=("random", "stated", "revealed", "reciprocal"), k=3):
    """How concentrated are recommendation slots on a few partners?"""
    rows = []
    for rule in rules:
        ranked = scored.sort_values(["iid", rule, "tiebreak"], ascending=[True, False, True])
        chosen = ranked.groupby("iid").head(k)
        ginis, shares = [], []
        for (wave, code), group in scored.groupby(["wave", "gender"]):
            counts = (chosen[(chosen["wave"] == wave) & (chosen["gender"] == code)]["pid"]
                      .value_counts().reindex(group["pid"].unique()).fillna(0))
            ginis.append(gini(counts))
            top = max(1, len(counts) // 5)
            shares.append(counts.sort_values(ascending=False).head(top).sum() / counts.sum())
        rows.append(dict(rule=RULE_LABELS[rule], gini=np.mean(ginis), top_20pct_share=np.mean(shares)))
    return pd.DataFrame(rows)


def deferred_acceptance(proposer_utility, receiver_utility):
    """Gale-Shapley stable matching. NaN means the pair is unavailable."""
    n_proposers, n_receivers = proposer_utility.shape
    order = [np.argsort(-np.nan_to_num(row, nan=-np.inf)) for row in proposer_utility]
    pointer = [0] * n_proposers
    free, held = list(range(n_proposers)), {}
    while free:
        proposer = free.pop()
        while pointer[proposer] < n_receivers:
            receiver = order[proposer][pointer[proposer]]
            pointer[proposer] += 1
            if not np.isfinite(proposer_utility[proposer, receiver]):
                continue
            if receiver not in held:
                held[receiver] = proposer
                break
            current = held[receiver]
            if receiver_utility[proposer, receiver] > receiver_utility[current, receiver]:
                free.append(current)
                held[receiver] = proposer
                break
    return [(proposer, receiver) for receiver, proposer in held.items()]


def history_candidates(ctx):
    """Later-evening pairs scored by models that only saw earlier dates."""
    later = add_pair_features(split_history(ctx.data))
    people = ctx.data.drop_duplicates("iid").set_index("iid")
    weights = people[STATED].astype(float)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(1 / 6)
    later["stated"] = (weights.reindex(later["iid"]).to_numpy() * later[HISTORY].to_numpy()).sum(axis=1)
    full, plain = HISTORY + ["h_rcv", "h_sel"] + PAIR, HISTORY + PAIR
    parts = []
    for wave in sorted(later["wave"].unique()):
        for code, _ in GENDERS:
            train = later[(later["wave"] != wave) & (later["gender"] == code)]
            test = later[(later["wave"] == wave) & (later["gender"] == code)].copy()
            if test.empty:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            test["p_yes"] = model.fit(train[full], train["dec"]).predict_proba(test[full])[:, 1]
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            test["revealed"] = model.fit(train[plain], train["dec"]).decision_function(test[plain])
            parts.append(test)
    scored = pd.concat(parts, ignore_index=True)
    other = scored[["iid", "pid", "p_yes", "stated", "revealed"]].rename(columns={
        "iid": "pid", "pid": "iid", "p_yes": "p_yes_other", "stated": "stated_other",
        "revealed": "revealed_other"})
    scored = scored.merge(other, on=["iid", "pid"], how="inner")
    scored["reciprocal"] = np.log(scored["p_yes"]) + np.log(scored["p_yes_other"])
    scored["tiebreak"] = np.random.default_rng(SEED).random(len(scored))
    scored["random"] = scored["tiebreak"]
    return scored


def simulate_market(scored, n_random=30):
    """Give each person one or two exclusive recommendations per event."""
    rows = []
    women_rows = scored[scored["gender"] == 0]
    for wave, event in women_rows.groupby("wave"):
        women, men = sorted(event["iid"].unique()), sorted(event["pid"].unique())
        w_index, m_index = {v: i for i, v in enumerate(women)}, {v: i for i, v in enumerate(men)}

        def matrix(column):
            out = np.full((len(men), len(women)), np.nan)
            out[event["pid"].map(m_index).to_numpy(), event["iid"].map(w_index).to_numpy()] = event[column]
            return out

        outcome = matrix("match")
        available = np.isfinite(outcome)
        joint = matrix("p_yes") * matrix("p_yes_other")

        def assign(score, used):
            values = np.where(np.isfinite(score) & ~used, score, -1e6)
            r, c = linear_sum_assignment(-values)
            keep = values[r, c] > -1e5
            return list(zip(r[keep], c[keep]))

        def rounds(policy, slots):
            used, pairs = ~available, []
            for _ in range(slots):
                chosen = policy(used)
                if not chosen:
                    break
                pairs += chosen
                for a, b in chosen:
                    used[a, b] = True
            return float(np.mean([outcome[a, b] for a, b in pairs]))

        def stable(men_score, women_score):
            return lambda used: deferred_acceptance(np.where(used, np.nan, men_score),
                                                    np.where(used, np.nan, women_score))

        if available.sum() < 4:
            continue
        for slots in (1, 2):
            rng = np.random.default_rng(SEED + int(wave))
            rows.append(dict(
                wave=wave, slots=slots, pairs=int(available.sum()),
                random=np.mean([rounds(lambda used: assign(rng.random(joint.shape), used), slots)
                                for _ in range(n_random)]),
                stated=rounds(stable(matrix("stated_other"), matrix("stated")), slots),
                revealed=rounds(stable(matrix("revealed_other"), matrix("revealed")), slots),
                optimised=rounds(lambda used: assign(joint, used), slots)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- A5 the metric


def a5_metric(ctx):
    """Does ranking on revealed weights raise matches, and what would?"""
    scored = ranking_frame(ctx)
    rules = ["random", "stated", "revealed", "revealed_pair", "popularity", "partner_only",
             "reciprocal", "post_stated", "post_revealed"]
    outcomes = {(rule, k): top_k(scored, rule, k) for rule in rules for k in (1, 3, 5)}
    labels = {**RULE_LABELS, "post_stated": "Stated, post-date ratings",
              "post_revealed": "Revealed, post-date ratings"}
    ranking = pd.DataFrame([
        dict(top_k=k, rule=labels[rule], yes_rate=v["dec"].mean(), match_rate=v["match"].mean(),
             partner_yes_rate=v["dec_o"].mean(), no_match_share=(v["match"] == 0).mean())
        for (rule, k), v in outcomes.items()])
    base = scored.groupby("iid")[["dec", "match", "dec_o"]].mean().mean()

    comparisons = [("revealed", "stated"), ("reciprocal", "stated"), ("reciprocal", "revealed"),
                   ("post_revealed", "post_stated")]
    lift_rows = []
    for k in (1, 3, 5):
        for first, second in comparisons:
            for column, metric in (("dec", "Yes rate"), ("match", "Match rate")):
                result = paired_lift(outcomes[first, k], outcomes[second, k], column, ctx.n_boot)
                lift_rows.append(dict(top_k=k, comparison=f"{labels[first]} minus {labels[second]}",
                                      metric=metric, lift_pp=result["estimate"],
                                      ci_low=result["ci_low"], ci_high=result["ci_high"], p=result["p"]))
    lifts = pd.DataFrame(lift_rows)

    seed_rows = []
    for seed in SEEDS:
        replicate = ranking_frame(ctx, seed) if seed != SEED else scored
        stated, revealed, both = (top_k(replicate, r, 3) for r in ("stated", "revealed", "reciprocal"))
        seed_rows.append(dict(seed=seed, stated_match=stated["match"].mean(),
                              revealed_match=revealed["match"].mean(),
                              reciprocal_match=both["match"].mean(),
                              reciprocal_lift_pp=(both["match"] - stated["match"]).mean() * 100,
                              reciprocal_yes_cost_pp=(both["dec"] - stated["dec"]).mean() * 100))
    seeds = pd.DataFrame(seed_rows)

    mode_rows = []
    for mode, label in [("full", "All partner facts"), ("no_selectivity", "Without partner yes rate"),
                        ("selectivity_only", "Only partner yes rate")]:
        replicate = scored if mode == "full" else ranking_frame(ctx, SEED, mode)
        lift = paired_lift(top_k(replicate, "reciprocal", 3), top_k(replicate, "stated", 3),
                           "match", ctx.n_boot)
        rng = np.random.default_rng(SEED)
        shuffled = replicate.copy()
        shuffled["partner_only"] = shuffled.groupby("iid")["partner_only"].transform(
            lambda v: rng.permutation(v.to_numpy()))
        shuffled["placebo"] = np.log(shuffled["own_side"]) + np.log(shuffled["partner_only"])
        placebo = top_k(shuffled, "placebo", 3)["match"].mean()
        mode_rows.append(dict(partner_model=label, match_lift_pp=lift["estimate"],
                              ci_low=lift["ci_low"], ci_high=lift["ci_high"], p=lift["p"],
                              placebo_match_rate=placebo))
    modes = pd.DataFrame(mode_rows)

    exposure = exposure_table(scored)

    desirability = scored.groupby("iid")["dec_o"].mean()
    tier = pd.qcut(desirability, 3, labels=["Least chosen third", "Middle third", "Most chosen third"])
    tier_rows = []
    for label in tier.cat.categories:
        ids = tier[tier == label].index
        for rule in ("stated", "revealed", "reciprocal"):
            chosen = outcomes[rule, 3].loc[ids]
            tier_rows.append(dict(tier=label, rule=RULE_LABELS[rule], people=len(ids),
                                  match_rate=chosen["match"].mean(),
                                  no_match_share=(chosen["match"] == 0).mean()))
    tiers = pd.DataFrame(tier_rows)

    history = history_candidates(ctx)
    history_tops = {(rule, k): top_k(history, rule, k)
                    for rule in ("random", "stated", "revealed", "reciprocal") for k in (1, 2, 3)}
    history_rows = []
    for k in (1, 2, 3):
        for first, second in [("reciprocal", "stated"), ("revealed", "stated")]:
            for column, metric in (("match", "Match rate"), ("dec", "Yes rate")):
                result = paired_lift(history_tops[first, k], history_tops[second, k], column, ctx.n_boot)
                history_rows.append(dict(
                    top_k=k, comparison=f"{RULE_LABELS[first]} minus {RULE_LABELS[second]}",
                    metric=metric, first=history_tops[first, k][column].mean(),
                    second=history_tops[second, k][column].mean(), lift_pp=result["estimate"],
                    ci_low=result["ci_low"], ci_high=result["ci_high"], p=result["p"]))
    history_table = pd.DataFrame(history_rows)

    people = ctx.data.groupby("iid").agg(
        dates=("dec", "size"), yeses=("dec", "sum"), yes_received=("dec_o", "sum"),
        matches=("match", "sum")).join(ctx.data.drop_duplicates("iid").set_index("iid")[
            ["date_3", "you_call", "them_cal", "gender", "satis_2", "numdat_3"]])
    answered = people[people["date_3"].notna()]
    some = answered[answered["matches"] > 0]
    date_model = smf.logit("date_3 ~ matches + C(gender)", some).fit(disp=0)
    margin = date_model.get_margeff().summary_frame().loc["matches"]
    people["answered_next_day"] = people["satis_2"].notna().astype(int)
    people["no_match"] = (people["matches"] == 0).astype(int)
    response = smf.logit("answered_next_day ~ no_match + C(gender)", people).fit(disp=0)
    funnel = pd.DataFrame([
        dict(stage="Speed dates per person", value=people["dates"].mean(), people=len(people)),
        dict(stage="Yeses given", value=people["yeses"].mean(), people=len(people)),
        dict(stage="Mutual matches", value=people["matches"].mean(), people=len(people)),
        dict(stage="Share with no match", value=people["no_match"].mean(), people=len(people)),
        dict(stage="Answered the 3-4 week follow-up", value=len(answered) / len(people), people=len(people)),
        dict(stage="Contacted a match", value=(answered["you_call"] > 0).mean(), people=len(answered)),
        dict(stage="Was contacted", value=(answered["them_cal"] > 0).mean(), people=len(answered)),
        dict(stage="Went on a date with a match", value=answered["date_3"].mean(), people=len(answered)),
        dict(stage="Date rate if at least one match", value=some["date_3"].mean(), people=len(some)),
        dict(stage="Change in date chance per extra match", value=margin["dy/dx"], people=len(some)),
        dict(stage="Next-day response, no match", value=people[people["no_match"] == 1]["answered_next_day"].mean(),
             people=int(people["no_match"].sum())),
        dict(stage="Next-day response, at least one match",
             value=people[people["no_match"] == 0]["answered_next_day"].mean(),
             people=int((people["no_match"] == 0).sum())),
    ])

    def lift(top, comparison, metric):
        return lifts[(lifts["top_k"] == top) & (lifts["comparison"] == comparison)
                     & (lifts["metric"] == metric)].iloc[0]

    def hist(top, comparison, metric):
        return history_table[(history_table["top_k"] == top) & (history_table["comparison"] == comparison)
                             & (history_table["metric"] == metric)].iloc[0]

    rev_label = "Revealed weights minus Stated weights"
    rec_label = "Both likely to say yes minus Stated weights"
    post_label = "Revealed, post-date ratings minus Stated, post-date ratings"
    rev_match = lift(3, rev_label, "Match rate")
    rev_yes = lift(1, rev_label, "Yes rate")
    rev_yes_3 = lift(3, rev_label, "Yes rate")
    post_match = lift(3, post_label, "Match rate")
    rec_match, rec_yes = lift(3, rec_label, "Match rate"), lift(3, rec_label, "Yes rate")
    strict_3, strict_1 = hist(3, rec_label, "Match rate"), hist(1, rec_label, "Match rate")
    stated_rows = ranking[ranking["rule"] == "Stated weights"].set_index("top_k")
    revealed_rows = ranking[ranking["rule"] == "Revealed weights"].set_index("top_k")
    both_rows = ranking[ranking["rule"] == "Both likely to say yes"].set_index("top_k")
    no_selectivity = modes[modes["partner_model"] == "Without partner yes rate"].iloc[0]
    n_scored = scored["iid"].nunique()
    low_stated = tiers[(tiers["tier"] == "Least chosen third") & (tiers["rule"] == "Stated weights")].iloc[0]
    low_both = tiers[(tiers["tier"] == "Least chosen third") & (tiers["rule"] == "Both likely to say yes")].iloc[0]
    rows = [
        finding("A5a", f"With pre-date information, top-1 yes is {stated_rows.loc[1, 'yes_rate']:.0%} "
                       f"(stated) and {revealed_rows.loc[1, 'yes_rate']:.0%} (revealed), against "
                       f"{base['dec']:.0%} on average. The gap between the two is not clear at top 1",
                rev_yes.lift_pp, (rev_yes.ci_low, rev_yes.ci_high), rev_yes.p, "pp yes, top 1",
                f"{n_scored} people", "a5_lifts", primary=True),
        finding("A5a2", f"At top 3, revealed ranking adds {rev_yes_3.lift_pp:+.1f} pp of yeses",
                rev_yes_3.lift_pp, (rev_yes_3.ci_low, rev_yes_3.ci_high), rev_yes_3.p, "pp yes, top 3",
                f"{n_scored} people", "a5_lifts", primary=True),
        finding("A5b", f"Revealed ranking does not raise matches: {rev_match.lift_pp:+.1f} pp at top 3",
                rev_match.lift_pp, (rev_match.ci_low, rev_match.ci_high), rev_match.p, "pp match, top 3",
                f"{n_scored} people", "a5_lifts", primary=True),
        finding("A5c", f"The same holds in the slide 8 simulation with post-date ratings: "
                       f"{post_match.lift_pp:+.1f} pp at top 3",
                post_match.lift_pp, (post_match.ci_low, post_match.ci_high), post_match.p,
                "pp match, top 3", f"{n_scored} people", "a5_lifts"),
        finding("A5d", f"Ranking for mutual interest lifts top-3 matches from "
                       f"{stated_rows.loc[3, 'match_rate']:.1%} to {both_rows.loc[3, 'match_rate']:.1%}, "
                       f"and changes yes rate by {rec_yes.lift_pp:+.1f} pp. This split is the highest of "
                       f"five: the lift ranges {seeds.reciprocal_lift_pp.min():+.1f} to "
                       f"{seeds.reciprocal_lift_pp.max():+.1f} pp",
                rec_match.lift_pp, (rec_match.ci_low, rec_match.ci_high), rec_match.p, "pp match, top 3",
                f"{n_scored} people, split-fold design", "a5_lifts", primary=True),
        finding("A5e", f"The stricter history design gives {strict_1.lift_pp:+.1f} pp at top 1 and "
                       f"{strict_3.lift_pp:+.1f} pp at top 3",
                strict_3.lift_pp, (strict_3.ci_low, strict_3.ci_high), strict_3.p, "pp match, top 3",
                f"{history['iid'].nunique()} people, later dates only", "a5_history_design", primary=True),
        finding("A5f", f"The lift comes from partner yes rate: without it the lift is "
                       f"{no_selectivity.match_lift_pp:+.1f} pp, so mass-likers would need a cap",
                no_selectivity.match_lift_pp, (no_selectivity.ci_low, no_selectivity.ci_high),
                no_selectivity.p, "pp match, top 3", f"{n_scored} people", "a5_partner_model"),
        finding("A5g", f"Among the least-chosen third, people with no top-3 match fall from "
                       f"{low_stated.no_match_share:.0%} to {low_both.no_match_share:.0%}",
                low_both.no_match_share - low_stated.no_match_share, unit="share of people",
                sample=f"{low_both.people} people", table="a5_who_benefits"),
        finding("A5h", f"Funnel: {people['dates'].mean():.1f} speed dates, {people['yeses'].mean():.1f} "
                       f"yeses, {people['matches'].mean():.1f} matches; {people['no_match'].mean():.0%} "
                       f"get none. Each extra match adds {margin['dy/dx'] * 100:.1f} pp to reporting a date",
                margin["dy/dx"] * 100, (margin.iloc[4] * 100, margin.iloc[5] * 100),
                margin["Pr(>|z|)"], "pp date chance per match", f"{len(some)} people with a match",
                "a5_funnel", primary=True),
        finding("A5i", f"People with no match answered the next-day survey less often "
                       f"({people[people['no_match'] == 1]['answered_next_day'].mean():.0%} vs "
                       f"{people[people['no_match'] == 0]['answered_next_day'].mean():.0%}), a weak "
                       f"stand-in for retention",
                response.params["no_match"], p=response.pvalues["no_match"], unit="log-odds",
                sample=f"{len(people)} people", table="a5_funnel", primary=True),
    ]
    return dict(tables=dict(a5_ranking=ranking, a5_lifts=lifts, a5_seed_sensitivity=seeds,
                            a5_partner_model=modes, a5_exposure=exposure, a5_who_benefits=tiers,
                            a5_history_design=history_table, a5_funnel=funnel), rows=rows)


# ------------------------------------------------------- S3 market simulator


def s3_market_simulator(ctx):
    """Do exclusive recommendations help, and which pairing rule is best?"""
    simulated = simulate_market(history_candidates(ctx))
    policies = ["random", "stated", "revealed", "optimised"]
    summary = simulated.groupby("slots")[policies].mean().reset_index()
    rows = []
    for slots, group in simulated.groupby("slots"):
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(group), (ctx.n_boot, len(group)))
        for first, second in [("optimised", "random"), ("stated", "random"), ("revealed", "random"),
                              ("optimised", "stated"), ("revealed", "stated")]:
            values = (group[first] - group[second]).to_numpy() * 100
            rows.append(dict(slots=slots, comparison=f"{first} minus {second}", lift_pp=values.mean(),
                             events=len(group), **summarise_draws(values[idx].mean(axis=1))))
    lifts = pd.DataFrame(rows)
    one = summary[summary["slots"] == 1].iloc[0]
    versus_random = lifts[(lifts["slots"] == 1) & (lifts["comparison"] == "optimised minus random")].iloc[0]
    versus_stated = lifts[(lifts["slots"] == 1) & (lifts["comparison"] == "optimised minus stated")].iloc[0]
    rows = [
        finding("S3a", f"One exclusive recommendation each: random {one.random:.1%}, stated "
                       f"{one.stated:.1%}, revealed {one.revealed:.1%}, optimised {one.optimised:.1%}",
                versus_random.lift_pp, (versus_random.ci_low, versus_random.ci_high), versus_random.p,
                "pp match, optimised vs random", f"{len(simulated) // 2} events", "s3_lifts", primary=True),
        finding("S3b", f"Optimised does not clearly beat stable matching on stated weights "
                       f"({versus_stated.lift_pp:+.1f} pp)",
                versus_stated.lift_pp, (versus_stated.ci_low, versus_stated.ci_high), versus_stated.p,
                "pp match", f"{len(simulated) // 2} events", "s3_lifts", primary=True),
    ]
    return dict(tables=dict(s3_market=summary, s3_lifts=lifts, s3_by_event=simulated), rows=rows)


# --------------------------------------------------- A3 survey vs behaviour


def a3_survey_vs_behaviour(ctx):
    """Predict later yeses from the survey, then from a few real interactions."""
    data = ctx.data
    later = data[data["order"] > 3].groupby("iid")["dec_o"].mean().rename("later_yes_rate")
    first = data[data["order"] <= 3].groupby("iid").agg(
        first_attractive=("attr_o", "mean"), first_yes=("dec_o", "mean"),
        first_fun=("fun_o", "mean"), first_shared=("shar_o", "mean"))
    people = data.drop_duplicates("iid").set_index("iid").join(later, how="inner").join(first)
    for column in ("income", "tuition", "mn_sat"):
        people[column] = pd.to_numeric(people[column].astype(str).str.replace(",", ""), errors="coerce")
    groups = {
        "Demographics": ["age", "income", "mn_sat", "tuition", "race", "field_cd", "career_c",
                         "imprace", "imprelig"],
        "Hobbies": HOBBIES,
        "Stated weights": STATED,
        "Self-ratings": ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"],
        "Lifestyle": ["goal", "date", "go_out", "exphappy"],
    }
    groups["Full sign-up survey"] = sum(groups.values(), [])
    groups["First 3 speed dates: yeses only"] = ["first_yes"]
    groups["First 3 speed dates: ratings and yeses"] = list(first.columns)

    rows, predictions = [], {}
    for code, _ in GENDERS:
        name = {0: "Women", 1: "Men"}[code]
        side = people[(people["gender"] == code) & people["later_yes_rate"].notna()]
        y = side["later_yes_rate"].to_numpy()

        def r2(predicted, target=y):
            return 1 - ((target - predicted)**2).sum() / ((target - target.mean())**2).sum()

        for label, columns in groups.items():
            x = side[columns].apply(pd.to_numeric, errors="coerce")
            ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  RidgeCV(alphas=np.logspace(0, 4, 20)))
            boost = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, max_depth=3,
                                                  random_state=SEED)
            cv = GroupKFold(7)
            p_ridge = cross_val_predict(ridge, x, y, groups=side["wave"], cv=cv)
            p_boost = cross_val_predict(boost, x, y, groups=side["wave"], cv=cv)
            predictions[name, label] = (p_ridge, p_boost, y)
            rows.append(dict(people_rated=name, predictors=label, people=len(side),
                             r2_ridge=r2(p_ridge), r2_boosting=r2(p_boost)))
    table = pd.DataFrame(rows)

    gap_rows = []
    for name in ("Women", "Men"):
        behaviour = predictions[name, "First 3 speed dates: ratings and yeses"]
        survey = predictions[name, "Full sign-up survey"]
        y = behaviour[2]
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(y), (ctx.n_boot, len(y)))

        def r2_many(predicted):
            sampled, target = predicted[idx], y[idx]
            total = ((target - target.mean(axis=1, keepdims=True))**2).sum(axis=1)
            return 1 - ((target - sampled)**2).sum(axis=1) / total

        draws = r2_many(behaviour[0]) - r2_many(survey[0])
        point = (1 - ((y - behaviour[0])**2).sum() / ((y - y.mean())**2).sum()
                 - (1 - ((y - survey[0])**2).sum() / ((y - y.mean())**2).sum()))
        gap_rows.append(dict(people_rated=name, r2_gap=point, **summarise_draws(draws)))
    gaps = pd.DataFrame(gap_rows)

    pre = pre_event_ranking(ctx)
    women = table[(table["people_rated"] == "Women")].set_index("predictors")
    men = table[(table["people_rated"] == "Men")].set_index("predictors")
    survey, behaviour = "Full sign-up survey", "First 3 speed dates: ratings and yeses"
    best = pre["table"].sort_values("auc", ascending=False).iloc[0]
    rows = [
        finding("A3a", f"Three speed dates beat the full survey: R² {women.loc[behaviour, 'r2_ridge']:.2f} "
                       f"vs {women.loc[survey, 'r2_ridge']:.2f} (women rated), "
                       f"{men.loc[behaviour, 'r2_ridge']:.2f} vs {men.loc[survey, 'r2_ridge']:.2f} (men rated)",
                gaps.r2_gap[0], (gaps.ci_low[0], gaps.ci_high[0]), gaps.p[0], "R² gap, women rated",
                f"{int(women.people.iloc[0])} women", "a3_prediction", primary=True),
        finding("A3b", "The same comparison for men", gaps.r2_gap[1], (gaps.ci_low[1], gaps.ci_high[1]),
                gaps.p[1], "R² gap, men rated", f"{int(men.people.iloc[0])} men", "a3_r2_gap", primary=True),
        finding("A3c", f"Pre-event facts barely rank pairs: AUC {best.auc:.2f}, top-decile match rate "
                       f"{best.top_decile_match_rate:.1%} against {best.base_rate:.1%}",
                best.top_decile_lift, (best.lift_ci_low, best.lift_ci_high), None, "lift, top decile",
                f"{int(best.pairs)} pairs, unseen events", "a3_pre_event"),
    ]
    return dict(tables=dict(a3_prediction=table, a3_r2_gap=gaps, a3_pre_event=pre["table"]), rows=rows)


def pre_event_ranking(ctx):
    """One record per pair, predicting match from sign-up facts on unseen events."""
    people = ctx.data.drop_duplicates("iid").set_index("iid")
    own = ["age", "imprace", "imprelig", "go_out", "date", "goal", "exphappy", *STATED,
           "attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1", *HOBBIES]
    pairs = ctx.data[ctx.data["gender"] == 0][["iid", "pid", "wave", "match", "int_corr", "samerace"]]
    pairs = pairs.dropna(subset=["pid"]).reset_index(drop=True)
    women = people.reindex(pairs["iid"])[own].add_prefix("w_").reset_index(drop=True)
    men = people.reindex(pairs["pid"])[own].add_prefix("m_").reset_index(drop=True)
    x = pd.concat([pairs[["int_corr", "samerace"]], women, men], axis=1).apply(pd.to_numeric, errors="coerce")
    x["agegap"] = (x["w_age"] - x["m_age"]).abs()
    for attribute in ("attr", "sinc", "intel", "fun", "amb"):
        x[f"w_fit_{attribute}"] = x[f"w_{attribute}1_1"] * x[f"m_{attribute}3_1"]
        x[f"m_fit_{attribute}"] = x[f"m_{attribute}1_1"] * x[f"w_{attribute}3_1"]
    y = pairs["match"].astype(int)
    logistic = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=2000))
    boosting = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_depth=3,
                                              random_state=SEED)
    cv = GroupKFold(pairs["wave"].nunique())
    rows = []
    for label, model, features in [("Pair facts only", logistic, x[["int_corr", "samerace", "agegap"]]),
                                   ("All pre-event, logistic", logistic, x),
                                   ("All pre-event, boosting", boosting, x)]:
        predicted = cross_val_predict(model, features, y, groups=pairs["wave"], cv=cv,
                                      method="predict_proba")[:, 1]
        frame = pd.DataFrame(dict(y=y, p=predicted, wave=pairs["wave"]))
        ranked = frame.sort_values("p", ascending=False)
        top = ranked.groupby("wave").apply(lambda g: pd.Series(dict(
            hits=g["y"].head(max(1, len(g) // 10)).sum(), taken=max(1, len(g) // 10),
            matches=g["y"].sum(), total=len(g))), include_groups=False)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(top), (ctx.n_boot, len(top)))
        hits, taken = top["hits"].to_numpy(), top["taken"].to_numpy()
        matches, total = top["matches"].to_numpy(), top["total"].to_numpy()
        draws = (hits[idx].sum(axis=1) / taken[idx].sum(axis=1)) / (matches[idx].sum(axis=1) / total[idx].sum(axis=1))
        rate, base = hits.sum() / taken.sum(), y.mean()
        rows.append(dict(model=label, pairs=len(y), auc=roc_auc_score(y, predicted),
                         top_decile_match_rate=rate, base_rate=base, top_decile_lift=rate / base,
                         lift_ci_low=np.quantile(draws, 0.025), lift_ci_high=np.quantile(draws, 0.975)))
    return dict(table=pd.DataFrame(rows))


# --------------------------------------------- A6 limit of personalisation


def _adjusted_r2(frame, columns):
    """Adjusted R² of a yes on sets of person dummies; residuals returned."""
    n = len(frame)
    rows, cols, offset = [], [], 0
    for column in columns:
        codes = frame[column].astype("category").cat.codes.to_numpy()
        rows.append(np.arange(n))
        cols.append(codes + offset)
        offset += codes.max() + 1
    x = sp.csr_matrix((np.ones(n * len(columns)), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, offset))
    y = frame["dec"].to_numpy(dtype=float)
    centred = y - y.mean()
    residual = centred - x @ lsqr(x, centred, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    parameters = offset - len(columns) + 1
    return 1 - (residual @ residual / (n - parameters)) / (centred @ centred / (n - 1)), residual


def a6_personalisation_limit(ctx):
    """How much of a yes comes from the chooser, the partner, and neither?"""
    data = add_consensus(ctx.data)
    rows, leave_out = [], []
    data["residual"] = np.nan
    for code, name in ((0, "Women choosing"), (1, "Men choosing")):
        side = data[data["gender"] == code].dropna(subset=["pid", "dec"])
        chooser, _ = _adjusted_r2(side, ["iid"])
        partner, _ = _adjusted_r2(side, ["pid"])
        both, residual = _adjusted_r2(side, ["iid", "pid"])
        data.loc[side.index, "residual"] = residual
        rows.append(dict(group=name, rows=len(side), chooser_share=chooser,
                         partner_share=both - chooser, partner_alone=partner, both=both,
                         unexplained_share=1 - both))
        for wave in side["wave"].unique():
            kept = side[side["wave"] != wave]
            leave_out.append(dict(group=name, unexplained=1 - _adjusted_r2(kept, ["iid", "pid"])[0]))
    table = pd.DataFrame(rows)
    spread = pd.DataFrame(leave_out).groupby("group")["unexplained"].agg(["min", "max"])
    table["unexplained_low"] = table["group"].map(spread["min"])
    table["unexplained_high"] = table["group"].map(spread["max"])

    pairs = add_pair_features(data.dropna(subset=["residual"]))
    for attribute, stated in zip(ATTRS, STATED):
        pairs[f"fit_{attribute}"] = pairs[stated] * pairs[f"c_{attribute}"]
    features = PAIR + [f"fit_{a}" for a in ATTRS]
    model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3, random_state=SEED)
    predicted = cross_val_predict(model, pairs[features], pairs["residual"], groups=pairs["wave"],
                                  cv=GroupKFold(5))
    y = pairs["residual"].to_numpy()
    held_out_r2 = 1 - ((y - predicted)**2).sum() / ((y - y.mean())**2).sum()
    mirrored = data[["iid", "pid", "residual"]].dropna().merge(
        data[["iid", "pid", "residual"]].dropna().rename(
            columns={"iid": "pid", "pid": "iid", "residual": "other_side"}), on=["iid", "pid"])
    table["pair_facts_held_out_r2"] = held_out_r2
    table["mutual_chemistry_correlation"] = mirrored["residual"].corr(mirrored["other_side"])
    rows = [finding("A6", f"The chooser explains {table.chooser_share.min():.0%}-{table.chooser_share.max():.0%} "
                          f"of a yes, the partner {table.partner_share.min():.0%}-{table.partner_share.max():.0%}, "
                          f"and {table.unexplained_share.min():.0%}-{table.unexplained_share.max():.0%} is "
                          f"unexplained. Pair facts add R²={held_out_r2:.3f} on unseen events",
                    table.unexplained_share.max(),
                    (float(spread["min"].min()), float(spread["max"].max())), None,
                    "unexplained share; range drops one event", "All 8,368 two-sided rows",
                    "a6_variance")]
    return dict(tables=dict(a6_variance=table), rows=rows)


# ----------------------------------------------------- S2 and N7 sequencing


def contrast_estimates(frame, rng=None):
    """Effect of the previous partner, in pp. Shuffles order if rng is given.

    Returns previous partner's attractiveness, the same with previous decision
    held fixed, and previous decision. A lagged outcome with person effects is
    negative by construction (about -1/(T-1)), which is why every estimate is
    judged against shuffled order rather than against zero.
    """
    data = frame.copy()
    data["sequence"] = rng.permutation(len(data)) if rng is not None else data["order"]
    data = data.sort_values(["iid", "sequence"], kind="stable")
    sd = data["c_attr"].std()
    previous = data.groupby("iid")
    data["previous_attr"] = previous["c_attr"].shift(1) / sd
    data["previous_yes"] = previous["dec"].shift(1)
    data["current_attr"] = data["c_attr"] / sd
    data = data.dropna(subset=["previous_attr", "previous_yes"])
    columns = ["dec", "previous_attr", "previous_yes", "current_attr", "sequence", *CONSENSUS[1:]]
    within = demean(data, columns).to_numpy(dtype=float)
    y, x = within[:, 0], within[:, 1:]
    short = np.delete(x, 1, axis=1)
    alone = np.linalg.lstsq(short, y, rcond=None)[0]
    joint = np.linalg.lstsq(x, y, rcond=None)[0]
    return np.array([alone[0], joint[0], joint[1]]) * 100


def s2_contrast(ctx):
    """Does an attractive previous partner lower the next yes?"""
    frame = add_consensus(ctx.data).dropna(subset=[*CONSENSUS, "order", "dec"])
    names = ["Previous partner's attractiveness", "Same, holding previous decision fixed",
             "Previous decision was yes (N7)"]
    rows, placebo_draws = [], None
    for label, side in [("Everyone", frame), ("Women choosing", frame[frame["gender"] == 0]),
                        ("Men choosing", frame[frame["gender"] == 1])]:
        real = contrast_estimates(side)
        rng = np.random.default_rng(SEED)
        placebo = np.array([contrast_estimates(side, rng) for _ in range(N_PERMUTATIONS)])
        placebo_draws = placebo[:, 0] if placebo_draws is None else placebo_draws
        centre = placebo.mean(axis=0)
        p = (np.abs(placebo - centre) >= np.abs(real - centre)).mean(axis=0)
        for j, name in enumerate(names):
            rows.append(dict(group=label, effect=name, estimate_pp=real[j], placebo_mean_pp=centre[j],
                             excess_pp=real[j] - centre[j], placebo_p=(p[j] * N_PERMUTATIONS + 1) / (N_PERMUTATIONS + 1),
                             rows=len(side)))
    table = pd.DataFrame(rows)

    data = frame.sort_values(["iid", "order"]).copy()
    sd = data["c_attr"].std()
    data["previous_attr"] = data.groupby("iid")["c_attr"].shift(1) / sd
    data["current_attr"] = data["c_attr"] / sd
    data = data.dropna(subset=["previous_attr", "attr"])
    within = demean(data, ["attr", "previous_attr", "current_attr", "order"]).assign(iid=data["iid"])
    rating = ols_cluster("attr ~ -1 + previous_attr + current_attr + order", within, within["iid"])
    table["rating_change_per_sd"] = rating.params["previous_attr"]
    table["rating_change_p"] = rating.pvalues["previous_attr"]

    everyone = table[(table["group"] == "Everyone")].set_index("effect")
    main, lagged = everyone.loc[names[0]], everyone.loc[names[2]]
    men = table[(table["group"] == "Men choosing") & (table["effect"] == names[0])].iloc[0]
    women = table[(table["group"] == "Women choosing") & (table["effect"] == names[0])].iloc[0]
    rows = [
        finding("S2", f"An attractive previous partner lowers the next yes by {-main.excess_pp:.1f} pp "
                      f"per SD (men {men.excess_pp:+.1f}, women {women.excess_pp:+.1f}); the person's "
                      f"rating of the next partner does not change",
                main.excess_pp, p=main.placebo_p, unit="pp per SD, beyond placebo",
                sample=f"{int(main.rows)} rows", table="s2_contrast", primary=True),
        finding("N7", f"'After a yes, the next yes drops' is an artefact: {lagged.estimate_pp:+.1f} pp "
                      f"against {lagged.placebo_mean_pp:+.1f} pp in shuffled order",
                lagged.excess_pp, p=lagged.placebo_p, unit="pp beyond placebo",
                sample=f"{int(lagged.rows)} rows", table="s2_contrast", primary=True),
    ]
    return dict(tables=dict(s2_contrast=table,
                            s2_placebo=pd.DataFrame(dict(shuffled_estimate_pp=placebo_draws))),
                rows=rows)


# ------------------------------------------------- N6 first-choices learning


def _personalise(x, y, prior, penalty):
    """Shift population weights towards a person's first choices."""
    def objective(theta):
        intercept, delta = theta[0], theta[1:]
        eta = intercept + x @ (prior + delta)
        probability = 1 / (1 + np.exp(-eta))
        loss = (np.logaddexp(0, eta) - y * eta).sum() + penalty * delta @ delta + 0.01 * intercept**2
        gradient = np.r_[(probability - y).sum() + 0.02 * intercept,
                         x.T @ (probability - y) + 2 * penalty * delta]
        return loss, gradient
    return minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B").x[1:]


def n6_first_choices(ctx):
    """Can a person's first few choices beat their survey for the rest of the night?"""
    frame = add_split_fold(ctx.data)
    sample = core_sample(frame).dropna(subset=[*SPLIT_FOLD, "order"])
    rows = []
    for wave in sorted(sample["wave"].unique()):
        for code, _ in GENDERS:
            train = sample[(sample["wave"] != wave) & (sample["gender"] == code)]
            test = sample[(sample["wave"] == wave) & (sample["gender"] == code)]
            if test.empty:
                continue
            fit = fit_clogit(make_strata(train, SPLIT_FOLD))
            for iid, person in test.groupby("iid"):
                person = person.sort_values("order")
                z = (person[SPLIT_FOLD].to_numpy(dtype=float) - fit.mean) / fit.sd
                decisions = person["dec"].to_numpy(dtype=float)
                stated = (person[STATED].to_numpy() * person[SPLIT_FOLD].to_numpy()).sum(axis=1)
                for seen in (3, 5, 8):
                    if len(person) - seen < 4:
                        continue

                    def top_three(score):
                        return decisions[seen:][np.argsort(-score)[:3]].mean()

                    delta = _personalise(z[:seen], decisions[:seen], fit.beta, 5.0)
                    rows.append(dict(iid=iid, choices_seen=seen, stated=top_three(stated[seen:]),
                                     population=top_three(z[seen:] @ fit.beta),
                                     personalised=top_three(z[seen:] @ (fit.beta + delta))))
    raw = pd.DataFrame(rows)
    table = []
    for seen, group in raw.groupby("choices_seen"):
        gain = boot_mean((group["personalised"] - group["population"]) * 100, ctx.n_boot, SEED)
        versus = boot_mean((group["personalised"] - group["stated"]) * 100, ctx.n_boot, SEED)
        table.append(dict(choices_seen=seen, people=len(group), stated=group["stated"].mean(),
                          population=group["population"].mean(), personalised=group["personalised"].mean(),
                          gain_over_population_pp=gain["estimate"], gain_ci_low=gain["ci_low"],
                          gain_ci_high=gain["ci_high"], gain_p=gain["p"],
                          gain_over_stated_pp=versus["estimate"], stated_ci_low=versus["ci_low"],
                          stated_ci_high=versus["ci_high"]))
    table = pd.DataFrame(table)
    middle = table[table["choices_seen"] == 5].iloc[0]
    rows = [finding("N6", f"Learning from the first 5 choices adds only {middle.gain_over_population_pp:+.1f} pp "
                          f"over population weights", middle.gain_over_population_pp,
                    (middle.gain_ci_low, middle.gain_ci_high), middle.gain_p, "pp yes, top 3",
                    f"{int(middle.people)} people", "n6_first_choices", primary=True)]
    return dict(tables=dict(n6_first_choices=table), rows=rows)


ANALYSES = [a3_survey_vs_behaviour, a5_metric, a6_personalisation_limit, s2_contrast,
            s3_market_simulator, n6_first_choices]
