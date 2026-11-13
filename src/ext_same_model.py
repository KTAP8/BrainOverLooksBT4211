"""A5o: one model for both sides of the match (final ranking rule for the deck).

Model A (conditional logit on the partner as rated by others in the opposite split
fold + pair facts, turned into a probability with the chooser's yes rate) predicts
"will you say yes?". The same model, fitted for the partner's gender on the other
events, is applied to the reverse row to predict "will they say yes?". The partner's
yes rate there is leave-one-out: it never contains their decision about you.

Compared on the same unseen-event replay as A5 (top 3, one model per held-out event
and gender), five split seeds:
  B0 mixed      A x the repo's model B (yes rate, stated weights, pair facts)
  both shared   A x logistic on the partner's yes rate + pair facts
  same model    A x A on the reverse row

Standalone: re-implements the A5 replay from ext_market.ranking_frame rather than
importing it. B0 reproduces ext_a5_seed_sensitivity.csv exactly (checked in main()).

Usage: python src/ext_same_model.py   (writes outputs/tables/ext_a5_same_model.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "clean_unscaled.csv"
OUT = ROOT / "outputs" / "tables" / "ext_a5_same_model.csv"
ATTRS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
STATED = [a + "1_1" for a in ATTRS]
PSTATED = ["pf_o_att", "pf_o_sin", "pf_o_int", "pf_o_fun", "pf_o_amb", "pf_o_sha"]
PAIR = ["int_corr", "agegap", "samerace"]
SF = [f"sf_{a}" for a in ATTRS]
SEEDS = [42, 7, 123, 2024, 99]
RULES = {"b0_mixed": "B0 mixed (repo headline)", "both_shared": "Both shared",
         "same_model": "Same model both ways"}


def logistic(x):
    return 1 / (1 + np.exp(-x))


def leave_one_out_rate(frame):
    """Each chooser's yes rate over their other rows at the event, clipped as in A5."""
    total = frame.groupby("iid")["dec"].transform("sum")
    count = frame.groupby("iid")["dec"].transform("count")
    rate = (total - frame["dec"]) / (count - 1)
    return rate.fillna(frame["dec"].mean()).clip(0.02, 0.98)


def own_side(frame, params, rate):
    """Model A as a probability: the chooser's yes rate shifted by centred utility."""
    utility = pd.Series(frame[SF + PAIR].to_numpy() @ params, index=frame.index)
    centred = utility - utility.groupby(frame["iid"]).transform("mean")
    return logistic(np.log(rate / (1 - rate)) + centred)


def prepare(seed):
    d = pd.read_csv(DATA)
    d["agegap"] = (d["age"] - d["age_o"]).abs()
    for column in PAIR:
        d[column] = d[column].fillna(d[column].median())
    people = d["iid"].unique()
    fold = pd.Series(np.random.default_rng(seed).integers(0, 2, len(people)), index=people)
    d["fold"] = d["iid"].map(fold)
    for column in [*ATTRS, "dec"]:
        m = d.groupby(["pid", "fold"])[column].mean().rename(f"sf_{column}").reset_index()
        m["fold"] = 1 - m["fold"]
        d = d.merge(m, on=["pid", "fold"], how="left")
    sides = d[["iid", "pid", "dec"]].assign(partner_fold=d["pid"].map(fold))
    sel = sides.groupby(["iid", "partner_fold"])["dec"].mean().rename("j_sel").reset_index()
    sel["fold"] = 1 - sel["partner_fold"]
    d = d.merge(sel.rename(columns={"iid": "pid"})[["pid", "fold", "j_sel"]], on=["pid", "fold"], how="left")
    d["j_sel"] = d["j_sel"].fillna(d["dec"].mean())
    parts = []
    for g in (0, 1):
        c = d[d["gender"] == g].dropna(subset=["iid", "gender", "wave", "dec"] + ATTRS + STATED)
        k = c.groupby("iid")["dec"].nunique()
        parts.append(c[c["iid"].isin(k[k == 2].index)])
    sample = pd.concat(parts).dropna(subset=SF).copy()
    sample["tiebreak"] = np.random.default_rng(seed).random(len(sample))
    return d, sample


def score(seed):
    d, s = prepare(seed)
    out, reverse = [], []
    for wave in sorted(s["wave"].unique()):
        for g in (0, 1):
            tr = s[(s["wave"] != wave) & (s["gender"] == g)]
            te = s[(s["wave"] == wave) & (s["gender"] == g)].copy()
            if te.empty:
                continue
            params = ConditionalLogit(tr["dec"].to_numpy(), tr[SF + PAIR].to_numpy(),
                                      groups=tr["iid"].to_numpy()).fit(disp=0).params
            # Your side, exactly as ext_market.ranking_frame (test-event yes rate).
            rate = te.groupby("iid")["dec"].transform("mean").clip(0.02, 0.98)
            te["own"] = own_side(te, params, rate)
            # Repo model B (B0).
            xb = pd.concat([te[PAIR], te[PSTATED].fillna(16.7) / 100, te[["j_sel"]]], axis=1).fillna(0)
            xb_tr = pd.concat([tr[PAIR], tr[PSTATED].fillna(16.7) / 100, tr[["j_sel"]]], axis=1).fillna(0)
            b0 = LogisticRegression(C=1.0, max_iter=1000).fit(xb_tr, tr["dec_o"]).predict_proba(xb)[:, 1]
            te["b0_mixed"] = np.log(te["own"]) + np.log(b0)
            # Shared weights on both sides: what varies by partner is their yes rate + pair facts.
            xs = PAIR + ["j_sel"]
            shared = LogisticRegression(C=1.0, max_iter=1000).fit(tr[xs], tr["dec_o"]).predict_proba(te[xs])[:, 1]
            te["both_shared"] = np.log(te["own"]) + np.log(shared)
            out.append(te)
            # Same model on the reverse rows: every chooser of this gender at the event,
            # leave-one-out yes rate so their decision about you never enters.
            everyone = d[(d["wave"] == wave) & (d["gender"] == g)].dropna(subset=SF + ["dec"]).copy()
            everyone["their_yes"] = own_side(everyone, params, leave_one_out_rate(everyone))
            reverse.append(everyone[["iid", "pid", "their_yes"]])
    scored = pd.concat(out)
    rev = pd.concat(reverse).drop_duplicates(["iid", "pid"]).rename(columns={"iid": "pid", "pid": "iid"})
    scored = scored.merge(rev, on=["iid", "pid"], how="left")
    missing = float(scored["their_yes"].isna().mean())
    scored["same_model"] = np.log(scored["own"]) + np.log(scored["their_yes"].fillna(scored["their_yes"].median()))
    return scored, missing


def top3(scored, rule):
    rows = []
    for _, grp in scored.groupby("iid"):
        top = grp.sort_values([rule, "tiebreak"], ascending=False).head(3)
        rows.append((top["match"].mean(), top["dec"].mean(), top["dec_o"].mean()))
    return np.array(rows)


def boot_ci(diff, draws=1000):
    means = [np.random.default_rng(i).choice(diff, len(diff)).mean() for i in range(draws)]
    return np.percentile(means, [2.5, 97.5])


def main():
    records = []
    for seed in SEEDS:
        scored, missing = score(seed)
        per = {rule: top3(scored, rule) for rule in RULES}
        for rule, label in RULES.items():
            row = {"seed": seed, "rule": label, "top3_match": per[rule][:, 0].mean(),
                   "top3_yes": per[rule][:, 1].mean(), "top3_partner_yes": per[rule][:, 2].mean(),
                   "people": len(per[rule]), "reverse_rows_missing": missing}
            for base in ("b0_mixed", "both_shared"):
                diff = per[rule][:, 0] - per[base][:, 0]
                low, high = boot_ci(diff)
                row[f"match_vs_{base}_pp"] = 100 * diff.mean()
                row[f"ci_low_vs_{base}"] = 100 * low
                row[f"ci_high_vs_{base}"] = 100 * high
            records.append(row)
        print(f"seed {seed}: " + ", ".join(
            f"{RULES[r]} {100 * per[r][:, 0].mean():.1f}%" for r in RULES))
    table = pd.DataFrame(records)
    # B0 must reproduce the repo's A5 headline replay.
    check = pd.read_csv(ROOT / "outputs" / "tables" / "ext_a5_seed_sensitivity.csv").set_index("seed")
    b0 = table[table["rule"] == RULES["b0_mixed"]].set_index("seed")["top3_match"]
    assert np.allclose(b0, check.loc[b0.index, "reciprocal_match"], atol=1e-6), "B0 does not reproduce A5"
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
