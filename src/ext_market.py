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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import ATTRS, SEED, STATED
from .ext_features import (CONSENSUS, GENDERS, HISTORY, PAIR, PARTNER_STATED, PERSONAL,
                           SPLIT_FOLD, YOU_SF, add_consensus, add_pair_features,
                           add_reverse_ratings, add_split_fold, boot_mean, core_sample, demean,
                           fit_clogit, learning_draws, make_strata, personal_weights, split_history,
                           summarise_draws)
from .ext_survey import finding, ols_cluster

N_PERMUTATIONS = 1000
SEEDS = (SEED, 7, 123, 2024, 99)
HOBBIES = ("sports tvsports exercise dining museums art hiking gaming clubbing reading tv "
           "theater movies concerts music shopping yoga").split()
RULE_LABELS = {"random": "Random order", "stated": "Stated weights",
               "revealed": "Revealed weights", "revealed_pair": "Revealed + pair facts",
               "popularity": "Popularity", "partner_only": "Partner likely to say yes",
               "reciprocal": "Both likely to say yes", "direct_match": "One model trained on matches"}


MIRRORS = {"mirror_stated": "fit_stated", "mirror_revealed": "fit_revealed",
           "full_fit_stated": "fit_stated", "full_fit_revealed": "fit_revealed"}
MIRROR_LABELS = {"full": "B0: current (yes rate, pair facts, stated weights)",
                 "mirror_stated": "B1: mirror, their stated weights x your ratings",
                 "mirror_revealed": "B2: mirror, their revealed weights x your ratings",
                 "full_fit_stated": "B0 + stated fit",
                 "full_fit_revealed": "B0 + revealed fit"}


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
    if mode.startswith("mirror"):
        # A5l: the mirror of model A. Their pickiness, the pair facts, and
        # how well you score on what this partner weighs.
        x["j_sel"] = frame["j_sel"]
        x["fit"] = frame[MIRRORS[mode]]
        return x.fillna(0)
    for attribute in ATTRS:
        x[attribute] = frame[PARTNER_STATED[attribute]].fillna(16.7) / 100
    if mode == "selectivity_only":
        x = x[[]].copy()
    if mode != "no_selectivity":
        x["j_sel"] = frame["j_sel"]
    if mode in MIRRORS:
        x["fit"] = frame[MIRRORS[mode]]
    return x.fillna(0)


def match_features(frame):
    """Every pre-date fact the two-sided rule uses, for one direct match model:
    the partner as others in the opposite fold saw them, the partner's yes rate,
    their stated weights and the pair facts."""
    x = partner_features(frame, "full")
    for column in SPLIT_FOLD:
        x[column] = frame[column].to_numpy()
    return x


def mirror_terms(sample, wave, code, lam=1.0, partner_n=None, draw=0):
    """Fit of each partner's revealed taste to the chooser's ratings (A5l).

    For choosers of gender `code`, partners are the other gender. Each
    partner's weights come from their own decisions on everyone except this
    chooser, shrunk towards pooled weights fitted without `wave`, so no
    decision or rating from the held-out event steers the training features.
    """
    other = 1 - code
    pool = sample[(sample["wave"] != wave) & (sample["gender"] == other)]
    prior = fit_clogit(make_strata(pool, SPLIT_FOLD))
    weights = personal_weights(sample[sample["gender"] == other], prior.beta, prior.mean,
                               prior.sd, lam, partner_n, seed=1000 * draw + wave)
    weights = weights.rename(columns={"iid": "pid", "pid": "iid"}).drop_duplicates(["iid", "pid"])
    rows = sample[sample["gender"] == code][["iid", "pid", *YOU_SF]].merge(
        weights, on=["iid", "pid"], how="left")
    for column, value in zip(PERSONAL, prior.beta):
        rows[column] = rows[column].fillna(value)
    you = (rows[YOU_SF].to_numpy(float) - prior.mean) / prior.sd
    return pd.Series((rows[PERSONAL].to_numpy() * you).sum(axis=1),
                     index=sample.index[sample["gender"] == code])


def ranking_frame(ctx, seed=SEED, mode="full", lam=1.0, shuffle_fit=False, partner_n=None,
                  draw=0):
    """Score each person's list on events the models never saw.

    Mirror modes (A5l) swap model B for the mirror of model A. `shuffle_fit`
    permutes the fit term within each test list: a placebo for its value.
    """
    frame = add_pair_features(add_split_fold(ctx.data, seed))
    sample = core_sample(frame).dropna(subset=SPLIT_FOLD).copy()
    sample["tiebreak"] = np.random.default_rng(seed).random(len(sample))
    sample["random"] = sample["tiebreak"]
    if mode in MIRRORS:
        reverse = add_reverse_ratings(frame)[["iid", "pid", *YOU_SF]].drop_duplicates(["iid", "pid"])
        sample = sample.merge(reverse, on=["iid", "pid"], how="left")
        stated = sample[[PARTNER_STATED[a] for a in ATTRS]].fillna(16.7).to_numpy() / 100
        sample["fit_stated"] = (stated * sample[YOU_SF].to_numpy()).sum(axis=1)
    parts = []
    for wave in sorted(sample["wave"].unique()):
        for code, _ in GENDERS:
            if MIRRORS.get(mode) == "fit_revealed":
                sample["fit_revealed"] = np.nan
                sample.loc[sample["gender"] == code, "fit_revealed"] = mirror_terms(
                    sample, wave, code, lam, partner_n, draw)
            train = sample[(sample["wave"] != wave) & (sample["gender"] == code)]
            test = sample[(sample["wave"] == wave) & (sample["gender"] == code)].copy()
            if test.empty:
                continue
            if shuffle_fit and mode in MIRRORS:
                rng = np.random.default_rng(seed + wave)
                test[MIRRORS[mode]] = test.groupby("iid")[MIRRORS[mode]].transform(
                    lambda v: rng.permutation(v.to_numpy()))
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
            # A5k: one model trained on the match itself, from the same facts.
            direct = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            direct.fit(match_features(train), train["match"])
            test["direct_match"] = direct.predict_proba(match_features(test))[:, 1]
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


ENDINGS = ["neither", "only_i_said_yes", "only_they_said_yes", "match"]


def endings(per_person):
    """Split slots into four endings from per-person means of dec, dec_o, match."""
    out = pd.DataFrame({"match": per_person["match"],
                        "only_i_said_yes": per_person["dec"] - per_person["match"],
                        "only_they_said_yes": per_person["dec_o"] - per_person["match"]})
    out["neither"] = 1 - out.sum(axis=1)
    return out[ENDINGS]


def endings_table(data, scored, outcomes):
    """A5j: how often interest goes unreturned, overall and in each top 3."""
    rows = []
    every = data.dropna(subset=["dec", "dec_o", "match"])
    for label, part in [("All dates", every), ("All dates: men choosing", every[every["gender"] == 1]),
                        ("All dates: women choosing", every[every["gender"] == 0])]:
        shares = endings(part[["dec", "dec_o", "match"]].astype(float)).mean()
        rows.append(dict(view=label, unit="dates", count=len(part), **shares))
    shares = endings(scored.groupby("iid")[["dec", "dec_o", "match"]].mean()).mean()
    rows.append(dict(view="Core sample, average person", unit="people",
                     count=scored["iid"].nunique(), **shares))
    for rule in ("random", "stated", "revealed", "reciprocal"):
        top = outcomes[rule, 3]
        rows.append(dict(view=f"Top 3: {RULE_LABELS[rule]}", unit="people", count=len(top),
                         **endings(top).mean()))
    table = pd.DataFrame(rows)
    table["one_sided"] = table["only_i_said_yes"] + table["only_they_said_yes"]
    table["unreturned_yes_share"] = table["only_i_said_yes"] / (table["only_i_said_yes"] + table["match"])
    return table


def _within_person(frame, score, target, metric):
    """Mean per-person score of how well `score` orders one person's list."""
    values = []
    for _, group in frame.groupby("iid"):
        y = group[target].to_numpy()
        if 0 < y.sum() < len(y):
            values.append(metric(y, group[score].to_numpy()))
    return np.array(values)


def rank_quality(scored):
    """How well each rule orders a person's list, for their yes and for a match.

    Scored within each person, since that is all a ranking uses; pooled
    metrics would reward knowing who says yes to everyone. Average precision is
    shown against the base rate of the same people, as the Kaggle notebook
    suggests for a rare outcome.
    """
    rows = []
    for target, label in (("match", "Mutual match"), ("dec", "Own yes")):
        eligible = scored.groupby("iid")[target].transform(lambda y: 0 < y.sum() < len(y)).astype(bool)
        base = scored[eligible].groupby("iid")[target].mean().mean()
        for rule in ("random", "stated", "revealed", "revealed_pair", "reciprocal", "direct_match"):
            auc = _within_person(scored, rule, target, roc_auc_score)
            precision = _within_person(scored, rule, target, average_precision_score)
            rows.append(dict(target=label, rule=RULE_LABELS[rule], people=len(auc),
                             roc_auc=auc.mean(), average_precision=precision.mean(), base_rate=base))
    return pd.DataFrame(rows)


def direct_vs_two_sided(scored, seed, n_boot):
    """A5k: one model trained on matches, against the two-sided rule."""
    rows = []
    for k in (1, 3):
        both, direct = top_k(scored, "reciprocal", k), top_k(scored, "direct_match", k)
        for column, metric in (("match", "Match rate"), ("dec", "Yes rate")):
            lift = paired_lift(both, direct, column, n_boot)
            rows.append(dict(seed=seed, top_k=k, metric=metric, two_sided=both[column].mean(),
                             direct_model=direct[column].mean(), two_sided_minus_direct_pp=lift["estimate"],
                             ci_low=lift["ci_low"], ci_high=lift["ci_high"], p=lift["p"],
                             people=len(both)))
    return rows


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

    seed_rows, direct_rows = [], []
    for seed in SEEDS:
        replicate = ranking_frame(ctx, seed) if seed != SEED else scored
        direct_rows += direct_vs_two_sided(replicate, seed, ctx.n_boot)
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
    endings_tbl = endings_table(ctx.data, scored, outcomes)
    ending = endings_tbl.set_index("view")
    unreturned = paired_lift(outcomes["reciprocal", 3].eval("dec - match").to_frame("x"),
                             outcomes["stated", 3].eval("dec - match").to_frame("x"), "x", ctx.n_boot)
    quality = rank_quality(scored)
    direct_tbl = pd.DataFrame(direct_rows)
    main_direct = direct_tbl[(direct_tbl["seed"] == SEED) & (direct_tbl["top_k"] == 3)
                             & (direct_tbl["metric"] == "Match rate")].iloc[0]
    direct_range = direct_tbl[(direct_tbl["top_k"] == 3) & (direct_tbl["metric"] == "Match rate")][
        "two_sided_minus_direct_pp"]
    auc = quality.set_index(["target", "rule"])["roc_auc"]
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
        finding("A5j", f"{ending.loc['All dates', 'one_sided']:.0%} of dates end one-sided and "
                       f"{ending.loc['All dates', 'unreturned_yes_share']:.0%} of yeses are never returned. "
                       f"In the top 3, ranking for both sides moves unreturned-yes slots from "
                       f"{ending.loc['Top 3: Stated weights', 'only_i_said_yes']:.0%} to "
                       f"{ending.loc['Top 3: Both likely to say yes', 'only_i_said_yes']:.0%}",
                unreturned["estimate"], (unreturned["ci_low"], unreturned["ci_high"]), unreturned["p"],
                "pp of top-3 slots with an unreturned yes", f"{n_scored} people", "a5_endings",
                primary=True),
        finding("A5k", f"Against one model trained on matches from the same facts, the two-sided rule "
                       f"gives {main_direct.two_sided_minus_direct_pp:+.1f} pp top-3 matches "
                       f"({direct_range.min():+.1f} to {direct_range.max():+.1f} across five splits). "
                       f"Within-list AUC: {auc['Mutual match', 'Both likely to say yes']:.2f} for "
                       f"matches (two-sided rule), {auc['Own yes', 'Revealed + pair facts']:.2f} for "
                       f"own yeses (revealed + pair facts)",
                main_direct.two_sided_minus_direct_pp, (main_direct.ci_low, main_direct.ci_high),
                main_direct.p, "pp match, top 3", f"{n_scored} people", "a5_direct_vs_two_sided",
                primary=True),
    ]
    return dict(tables=dict(a5_ranking=ranking, a5_lifts=lifts, a5_seed_sensitivity=seeds,
                            a5_partner_model=modes, a5_exposure=exposure, a5_who_benefits=tiers,
                            a5_history_design=history_table, a5_funnel=funnel,
                            a5_endings=endings_tbl, a5_direct_vs_two_sided=direct_tbl,
                            a5_rank_quality=quality), rows=rows)


def a5l_partner_mirror(ctx):
    """A5l: does model B gain from checking whether you meet the partner's taste?

    B0 is the published partner model. B1 and B2 mirror model A: the
    partner's yes rate, pair facts, and a fit score of their weights (stated,
    or revealed from their other dates) times how others rated you. A side
    test: the headline A5 numbers stay on B0.
    """
    modes = ["full", "mirror_stated", "mirror_revealed", "full_fit_stated", "full_fit_revealed"]
    rows, scored_main = [], {}
    for seed in SEEDS:
        frames = {mode: ranking_frame(ctx, seed, mode) for mode in modes}
        base = top_k(frames["full"], "reciprocal", 3)
        for mode, scored in frames.items():
            top = top_k(scored, "reciprocal", 3)
            match = paired_lift(top, base, "match", ctx.n_boot)
            yes = paired_lift(top, base, "dec", ctx.n_boot)
            rows.append(dict(seed=seed, partner_model=MIRROR_LABELS[mode], lam=1.0,
                             top3_match=top["match"].mean(), top3_yes=top["dec"].mean(),
                             partner_yes_auc=_within_person(scored, "partner_only", "dec_o",
                                                            roc_auc_score).mean(),
                             match_vs_b0_pp=match["estimate"], ci_low=match["ci_low"],
                             ci_high=match["ci_high"], p=match["p"],
                             yes_vs_b0_pp=yes["estimate"], people=len(top)))
        if seed == SEED:
            scored_main = frames
    base = top_k(scored_main["full"], "reciprocal", 3)
    extra = [(mode, 1.0, True) for mode in modes[1:]]
    extra += [("mirror_revealed", lam, False) for lam in (0.3, 3.0)]
    for mode, lam, shuffle in extra:
        scored = ranking_frame(ctx, SEED, mode, lam=lam, shuffle_fit=shuffle)
        top = top_k(scored, "reciprocal", 3)
        match = paired_lift(top, base, "match", ctx.n_boot)
        label = MIRROR_LABELS[mode] + (" [fit shuffled: placebo]" if shuffle else "")
        rows.append(dict(seed=SEED, partner_model=label, lam=lam, top3_match=top["match"].mean(),
                         top3_yes=top["dec"].mean(),
                         partner_yes_auc=_within_person(scored, "partner_only", "dec_o",
                                                        roc_auc_score).mean(),
                         match_vs_b0_pp=match["estimate"], ci_low=match["ci_low"],
                         ci_high=match["ci_high"], p=match["p"], yes_vs_b0_pp=np.nan,
                         people=len(top)))
    table = pd.DataFrame(rows)
    main = table[(table["seed"] == SEED) & (table["lam"] == 1.0)].set_index("partner_model")
    spread = table[table["lam"] == 1.0].groupby("partner_model")["match_vs_b0_pp"].agg(
        ["mean", "min", "max"])
    b0, b1, b2 = (main.loc[MIRROR_LABELS[m]] for m in ("full", "mirror_stated", "mirror_revealed"))
    s1, s2 = (spread.loc[MIRROR_LABELS[m]] for m in ("mirror_stated", "mirror_revealed"))
    row = finding(
        "A5l", f"Mirroring model A in model B (does the partner's taste fit how others rate you?) "
               f"adds about a point at most. Top-3 matches vs the current B, mean over five splits: "
               f"{s1['mean']:+.1f} pp with stated weights ({s1['min']:+.1f} to {s1['max']:+.1f}), "
               f"{s2['mean']:+.1f} pp with revealed weights ({s2['min']:+.1f} to {s2['max']:+.1f}). "
               f"Partner-yes AUC {b0.partner_yes_auc:.2f} / {b1.partner_yes_auc:.2f} / "
               f"{b2.partner_yes_auc:.2f}: one person's ~15 dates are too few to beat their stated "
               f"weights. Headline A5 stays on the current B",
        s1["mean"], (s1["min"], s1["max"]), None, "pp match, top 3, B1 minus B0 (range = splits)",
        f"{int(b1.people)} people", "a5_partner_mirror")
    return dict(tables=dict(a5_partner_mirror=table), rows=[row])


LEARNING_GRID = (2, 3, 4, 6, 8, 10, 12)
LEARNING_MIN_DATES = 16
LEARNING_TEST = 4
LEARNING_VERSIONS = {"Pre-date (others' ratings)": SPLIT_FOLD,
                     "Post-date (own ratings, as in 8a)": list(ATTRS)}


def learning_curve_frame(ctx, draws=50, lam=1.0, grid=LEARNING_GRID):
    """Per-chooser AUC on held-out dates, by weight source and n training dates.

    Choosers with at least LEARNING_MIN_DATES usable dates, so every n uses
    the same people and the same held-out dates. Stated: their own sign-up
    points on raw ratings. Pooled: everyone's revealed weights from the other
    events. Personal: weights from n of their other dates, shrunk towards
    pooled (`lam`), or almost unshrunk.
    """
    sample = add_split_fold(ctx.data).merge(ctx.sample[["iid", "pid"]].drop_duplicates(),
                                             on=["iid", "pid"])
    rows = []
    for version, columns in LEARNING_VERSIONS.items():
        usable = sample.dropna(subset=[*columns, "dec", *STATED])
        counts = usable.groupby("iid")["dec"].transform("size")
        usable = usable[counts >= LEARNING_MIN_DATES]
        for code, _ in GENDERS:
            for wave in sorted(usable["wave"].unique()):
                test = usable[(usable["wave"] == wave) & (usable["gender"] == code)]
                if test.empty:
                    continue
                pool = sample[(sample["wave"] != wave) & (sample["gender"] == code)]
                prior = fit_clogit(make_strata(pool, columns))
                for chooser, group in test.groupby("iid"):
                    y = group["dec"].to_numpy(float)
                    if not 0 < y.sum() < len(y):
                        continue
                    raw = group[columns].to_numpy(float)
                    x = np.ascontiguousarray((raw - prior.mean) / prior.sd)
                    stated = np.ascontiguousarray(raw @ group[STATED].iloc[0].to_numpy(float))
                    base = dict(version=version, iid=chooser, dates=len(y))
                    for shrink, line in ((lam, "Personal revealed"),
                                         (1e-2, "Personal revealed, unshrunk")):
                        for n in grid:
                            auc = learning_draws(x, y, stated, prior.beta, shrink, n,
                                                 LEARNING_TEST, draws, int(chooser) * 1000)
                            mean = np.nanmean(auc, axis=0)
                            rows.append(dict(base, line=line, n=n, auc=mean[0]))
                            if line == "Personal revealed":
                                rows.append(dict(base, line="Stated (their survey)", n=n,
                                                 auc=mean[1]))
                                rows.append(dict(base, line="Pooled revealed (everyone)", n=n,
                                                 auc=mean[2]))
    return pd.DataFrame(rows)


def a5m_learning_curve(ctx):
    """A5m: does a person's own revealed taste overtake their survey with more dates?"""
    people = learning_curve_frame(ctx)
    curve_rows, projection_rows, finding_rows = [], [], []
    for version, frame in people.groupby("version", sort=False):
        frame = frame.dropna(subset=["auc"])
        ids = sorted(set(frame["iid"]))
        for line, part in frame.groupby("line", sort=False):
            for n, cell in part.groupby("n"):
                cell = cell.set_index("iid")["auc"]
                stated_n = frame[(frame["line"] == "Stated (their survey)") & (frame["n"] == n)]
                level = boot_mean(cell.to_numpy(), ctx.n_boot)
                gap = boot_mean((cell - stated_n.set_index("iid")["auc"].reindex(cell.index))
                                .dropna().to_numpy() * 100, ctx.n_boot)
                curve_rows.append(dict(version=version, line=line, n=int(n),
                                       auc=cell.mean(), ci_low=level["ci_low"],
                                       ci_high=level["ci_high"], minus_stated_pp=gap["estimate"],
                                       gap_ci_low=gap["ci_low"], gap_ci_high=gap["ci_high"],
                                       gap_p=gap["p"], people=len(cell)))
        curve = pd.DataFrame([r for r in curve_rows if r["version"] == version])
        line = lambda name: curve[curve["line"] == name].sort_values("n")
        personal, raw = line("Personal revealed"), line("Personal revealed, unshrunk")
        stated = line("Stated (their survey)")["auc"].iloc[0]
        pooled_row = line("Pooled revealed (everyone)").iloc[0]
        # Linear trend of the unshrunk curve: where would it reach the survey?
        slope = np.polyfit(raw["n"], raw["auc"], 1)[0]
        reach = (raw["n"].iloc[-1] + (stated - raw["auc"].iloc[-1]) / slope
                 if slope > 0 and raw["auc"].iloc[-1] < stated else np.nan)
        projection_rows.append(dict(
            version=version, stated_auc=stated, pooled_auc=pooled_row.auc,
            pooled_minus_stated_pp=pooled_row.minus_stated_pp,
            personal_auc_first=personal["auc"].iloc[0], personal_auc_last=personal["auc"].iloc[-1],
            unshrunk_auc_first=raw["auc"].iloc[0], unshrunk_auc_last=raw["auc"].iloc[-1],
            unshrunk_gain_per_date=slope, unshrunk_reaches_survey_at_dates=reach,
            max_training_dates=int(raw["n"].iloc[-1]), people=len(ids)))
        first, last = personal.iloc[0], personal.iloc[-1]
        finding_rows.append(finding(
            "A5m" + ("a" if version.startswith("Pre") else "b"),
            f"{version}: everyone's revealed weights predict a person's own held-out choices better "
            f"than that person's survey (AUC {pooled_row.auc:.2f} vs {stated:.2f}). Weights learned "
            f"from the person's own dates do not catch up within {int(last.n)} dates: shrunk, they "
            f"slide from {first.auc:.2f} (near pooled) to {last.auc:.2f}; unshrunk they stay at "
            f"{raw['auc'].iloc[0]:.2f}-{raw['auc'].max():.2f}",
            pooled_row.minus_stated_pp, (pooled_row.gap_ci_low, pooled_row.gap_ci_high),
            pooled_row.gap_p, "AUC pts, pooled revealed minus own survey", f"{len(ids)} people",
            "a5_learning_curve"))
    return dict(tables=dict(a5_learning_curve=pd.DataFrame(curve_rows),
                            a5_learning_projection=pd.DataFrame(projection_rows)),
                rows=finding_rows)


PARTNER_GRID = (2, 4, 6, 8, 10, 12)


def a5n_partner_learning(ctx, seeds=SEEDS[:3], draws=3, grid=PARTNER_GRID):
    """A5n: does B2 (partner's revealed taste) catch B1 (stated) with more of their dates?

    Model A stays pooled. B2 learns each partner's weights from n of their
    other dates (never the date with this chooser). Outcome: top-3 match
    rate, averaged per person over splits and draws, compared with B1 on the
    same people. "Full-data" lists are those whose median partner has more
    than max(grid) dates, so n is a real n for most partners on the list (a
    stricter "every partner" rule leaves about 20 people).
    """
    person, auc_rows = {}, []

    def collect(key, scored):
        top = top_k(scored, "reciprocal", 3)["match"]
        person.setdefault(key, []).append(top)
        auc_rows.append(dict(key=key, auc=_within_person(scored, "partner_only", "dec_o",
                                                         roc_auc_score).mean()))
        return scored

    for seed in seeds:
        base = collect(("B0", None, None), ranking_frame(ctx, seed, "full"))
        collect(("B1", None, None), ranking_frame(ctx, seed, "mirror_stated"))
        for lam in (1.0, 0.3):
            for n in grid:
                for draw in range(draws):
                    collect(("B2", lam, n), ranking_frame(ctx, seed, "mirror_revealed", lam=lam,
                                                          partner_n=n, draw=draw))
    dates = base.groupby("iid").size()
    full = base.groupby("iid")["pid"].agg(lambda p: dates.reindex(p).fillna(0).median() > max(grid))
    full_ids = full[full].index
    means = {key: pd.concat(values, axis=1).mean(axis=1) for key, values in person.items()}
    auc = pd.DataFrame(auc_rows).groupby("key")["auc"].mean()
    rows = []
    for subset, ids in (("All lists", means[("B0", None, None)].index), ("Full-data lists", full_ids)):
        b1 = means[("B1", None, None)].reindex(ids)
        for key, values in means.items():
            values = values.reindex(ids)
            gap = boot_mean((values - b1).to_numpy() * 100, ctx.n_boot)
            label = ({"B0": "Current B", "B1": "B1: stated fit"}[key[0]] if key[0] != "B2"
                     else f"B2: revealed fit, shrinkage {key[1]:g}")
            rows.append(dict(subset=subset, model=label, partner_dates=key[2], lam=key[1],
                             top3_match=values.mean(), minus_b1_pp=gap["estimate"],
                             ci_low=gap["ci_low"], ci_high=gap["ci_high"], p=gap["p"],
                             partner_yes_auc=auc[key], people=len(ids)))
    table = pd.DataFrame(rows)
    findings = []
    for subset in ("Full-data lists", "All lists"):
        part = table[(table["subset"] == subset) & (table["lam"] == 1.0)].sort_values("partner_dates")
        last = part.iloc[-1]
        worst = part.loc[part["minus_b1_pp"].idxmin()]
        findings.append(finding(
            "A5n" + ("a" if subset.startswith("Full") else "b"),
            f"{subset}: learning each partner's taste from more of their dates does not catch their "
            f"survey. B2 minus B1 in top-3 matches is {part['minus_b1_pp'].iloc[0]:+.1f} pp at "
            f"{int(part['partner_dates'].iloc[0])} dates, {worst.minus_b1_pp:+.1f} at "
            f"{int(worst.partner_dates)}, {last.minus_b1_pp:+.1f} at {int(last.partner_dates)}: "
            f"below zero at every n, with no steady trend",
            last.minus_b1_pp, (last.ci_low, last.ci_high), last.p,
            f"pp match, top 3, B2 ({int(last.partner_dates)} dates) minus B1",
            f"{int(last.people)} people", "a5_partner_learning"))
    return dict(tables=dict(a5_partner_learning=table), rows=findings)


# ------------------------------------------------------- S3 market simulator


# ------------------------------------------------------ S4 common ground

COMMON_GROUND = [
    ("shar", "Felt shared interests (rated after the date)", [-0.1, 3, 5, 7, 10], ["0-3", "3-5", "5-7", "7-10"]),
    ("int_corr", "Actual hobby overlap (measured before)", None, ["Q1 least", "Q2", "Q3", "Q4 most"]),
    ("agegap", "Age gap (years)", [-0.1, 2, 5, 10, 60], ["0-2", "3-5", "6-10", "Over 10"]),
    ("samerace", "Same race", [-0.1, 0.5, 1.1], ["Different", "Same"]),
]


def cluster_rate(frame, column, n_boot, seed=SEED):
    """Mean of a 0/1 column with an interval that resamples people, not rows."""
    per = frame.groupby("iid")[column].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), (n_boot, len(per)))
    sums, counts = per["sum"].to_numpy()[idx].sum(axis=1), per["count"].to_numpy()[idx].sum(axis=1)
    draws = sums / counts
    return float(frame[column].mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def s4_common_ground(ctx):
    """Does feeling you share interests matter, or actually sharing them?

    Bands follow the Kaggle notebook's charts, on our real core sample. Slopes
    are per SD, with person-clustered errors, raw and within each rater
    (subtracting a rater's mean removes how readily they say yes overall).
    """
    frame = ctx.sample.copy()
    frame["agegap"] = (frame["age"] - frame["age_o"]).abs()
    band_rows, slope_rows = [], []
    for column, label, edges, names in COMMON_GROUND:
        part = frame.dropna(subset=[column]).copy()
        if edges is None:
            part["band"] = pd.qcut(part[column], 4, labels=names)
        else:
            part["band"] = pd.cut(part[column], edges, labels=names)
        for band, group in part.groupby("band", observed=True):
            match, m_low, m_high = cluster_rate(group, "match", ctx.n_boot)
            yes, y_low, y_high = cluster_rate(group, "dec", ctx.n_boot)
            band_rows.append(dict(measure=label, band=str(band), dates=len(group),
                                  people=group["iid"].nunique(), match_rate=match, match_ci_low=m_low,
                                  match_ci_high=m_high, yes_rate=yes, yes_ci_low=y_low,
                                  yes_ci_high=y_high, average_match_rate=part["match"].mean()))
        part["z"] = (part[column] - part[column].mean()) / part[column].std()
        for outcome in ("match", "dec"):
            for model in ("Raw", "Within rater"):
                data = part[["iid", "z", outcome]].copy()
                if model == "Within rater":
                    data[["z", outcome]] = demean(data, ["z", outcome])
                fit = ols_cluster(f"{outcome} ~ z", data, data["iid"])
                low, high = fit.conf_int().loc["z"]
                slope_rows.append(dict(measure=label, outcome="Mutual match" if outcome == "match" else "Own yes",
                                       model=model, pp_per_sd=fit.params["z"] * 100, ci_low=low * 100,
                                       ci_high=high * 100, p=fit.pvalues["z"], dates=len(data)))
    bands, slopes = pd.DataFrame(band_rows), pd.DataFrame(slope_rows)

    def band(measure, name):
        return bands[(bands["measure"] == measure) & (bands["band"] == name)].iloc[0]

    def slope(measure):
        return slopes[(slopes["measure"] == measure) & (slopes["outcome"] == "Mutual match")
                      & (slopes["model"] == "Within rater")].iloc[0]

    felt, actual = COMMON_GROUND[0][1], COMMON_GROUND[1][1]
    s_felt, s_actual = slope(felt), slope(actual)
    rows = [
        finding("S4a", f"Felt shared interests: match rate {band(felt, '0-3').match_rate:.0%} to "
                       f"{band(felt, '7-10').match_rate:.0%} from lowest to highest band",
                s_felt.pp_per_sd, (s_felt.ci_low, s_felt.ci_high), s_felt.p, "pp match per SD, within rater",
                f"{int(s_felt.dates)} dates", "s4_common_ground", primary=True),
        finding("S4b", f"Actual hobby overlap: match rate {band(actual, 'Q1 least').match_rate:.0%} to "
                       f"{band(actual, 'Q4 most').match_rate:.0%} from lowest to highest quarter",
                s_actual.pp_per_sd, (s_actual.ci_low, s_actual.ci_high), s_actual.p,
                "pp match per SD, within rater", f"{int(s_actual.dates)} dates", "s4_common_ground",
                primary=True),
    ]
    return dict(tables=dict(s4_common_ground=bands, s4_common_ground_slopes=slopes), rows=rows)


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


ANALYSES = [a3_survey_vs_behaviour, a5_metric, a5l_partner_mirror, a5m_learning_curve, a5n_partner_learning, a6_personalisation_limit, s2_contrast,
            s3_market_simulator, s4_common_ground, n6_first_choices]
