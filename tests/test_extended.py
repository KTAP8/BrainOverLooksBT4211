"""Checks for the extended analyses. Each one can fail for a real reason."""
import unittest

import numpy as np
import pandas as pd

from src import ext_features as ef, ext_market as em, ext_survey as es
from src.constants import ATTRS, TABLES_DIR
from src.io import load_clean, project_root


class Shared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_clean()
        cls.sample = ef.core_sample(cls.data)


class SampleAndLikelihood(Shared):
    def test_sample_matches_part3(self):
        flow = pd.read_csv(project_root() / TABLES_DIR / "core_model_sample_flow.csv")
        self.assertEqual(len(self.sample), int(flow["model_rows"].sum()))
        self.assertEqual(self.sample["iid"].nunique(), int(flow["model_people"].sum()))

    def test_sample_assertion_fires_on_altered_data(self):
        with self.assertRaises(ValueError):
            ef.core_sample(self.data[self.data["wave"] != 1])

    def test_fit_reproduces_part3_coefficients(self):
        """fit_clogit must agree with the published Part 3 model."""
        published = pd.read_csv(project_root() / TABLES_DIR / "core_model_results.csv")
        for code, name in ef.GENDERS:
            strata = ef.make_strata(self.sample[self.sample["gender"] == code], ATTRS)
            fit = ef.fit_clogit(strata)
            expected = published[published["gender"] == name].set_index("attribute")
            np.testing.assert_allclose(fit.beta, expected.loc[ATTRS, "coefficient_std"],
                                       atol=1e-3)

    def test_person_loglik_sums_to_model_loglik(self):
        strata = ef.make_strata(self.sample[self.sample["gender"] == 0], ATTRS)
        fit = ef.fit_clogit(strata)
        x = np.ascontiguousarray(strata.centred / fit.sd)
        total = ef.person_loglik(fit.beta, x, strata.y, strata.offsets, strata.yes_counts).sum()
        self.assertAlmostEqual(total, fit.loglike, places=4)


class Leakage(Shared):
    def test_loo_consensus_really_does_encode_the_rating(self):
        """Documents why ranking on it is banned: within a target, it falls
        as this rater's own rating rises."""
        frame = ef.add_consensus(self.data).dropna(subset=["attr", "c_attr"])
        within_rating = frame["attr"] - frame.groupby("pid")["attr"].transform("mean")
        within_consensus = frame["c_attr"] - frame.groupby("pid")["c_attr"].transform("mean")
        self.assertLess(within_rating.corr(within_consensus), -0.9)

    def test_split_fold_excludes_the_raters_own_fold(self):
        frame = ef.add_split_fold(self.data)
        row = frame.dropna(subset=["sf_attr"]).iloc[0]
        raters = frame[(frame["pid"] == row["pid"]) & (frame["fold"] != row["fold"])]
        self.assertAlmostEqual(row["sf_attr"], raters["attr"].mean())
        self.assertNotIn(row["iid"], set(raters["iid"]))

    def test_split_fold_is_constant_within_a_list(self):
        """If it varied, ranking across partners could exploit fold noise."""
        frame = ef.add_split_fold(self.data)
        self.assertEqual(frame.groupby("iid")["fold"].nunique().max(), 1)

    def test_changing_one_decision_leaves_its_own_features_alone(self):
        first = ef.add_split_fold(self.data).sort_values(["iid", "pid"]).reset_index(drop=True)
        altered = self.data.copy()
        row = altered.index[100]
        altered.loc[row, "dec"] = 1 - altered.loc[row, "dec"]
        second = ef.add_split_fold(altered).sort_values(["iid", "pid"]).reset_index(drop=True)
        key = (self.data.loc[row, "iid"], self.data.loc[row, "pid"])
        mask = (first["iid"] == key[0]) & (first["pid"] == key[1])
        self.assertEqual(first.loc[mask, "sf_dec"].item(), second.loc[mask, "sf_dec"].item())
        self.assertEqual(first.loc[mask, "j_sel"].item(), second.loc[mask, "j_sel"].item())

    def test_history_features_only_use_early_dates(self):
        later = ef.split_history(self.data)
        cut = self.data.groupby("wave")["order"].transform(lambda s: s.max() / 2)
        early = self.data[self.data["order"] <= cut]
        person = later["iid"].iloc[0]
        self.assertAlmostEqual(later[later["iid"] == person]["h_sel"].iloc[0],
                               early[early["iid"] == person]["dec"].mean())
        self.assertFalse(later.merge(early[["iid", "pid"]], on=["iid", "pid"]).shape[0])


class Market(Shared):
    def test_stable_matching_leaves_no_blocking_pair(self):
        rng = np.random.default_rng(3)
        men, women = rng.random((8, 8)), rng.random((8, 8))
        pairs = em.deferred_acceptance(men, women)
        self.assertEqual(len(pairs), 8)
        partner_of_man = dict(pairs)
        partner_of_woman = {w: m for m, w in pairs}
        for m in range(8):
            for w in range(8):
                prefers = men[m, w] > men[m, partner_of_man[m]]
                accepted = women[m, w] > women[partner_of_woman[w], w]
                self.assertFalse(prefers and accepted)

    def test_stable_matching_skips_unavailable_pairs(self):
        men = np.array([[1.0, np.nan], [np.nan, 1.0]])
        self.assertEqual(sorted(em.deferred_acceptance(men, men.copy())), [(0, 0), (1, 1)])

    def test_random_rule_recovers_the_base_rate(self):
        """A rule that knows nothing must not beat the average."""
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(dict(iid=np.repeat(np.arange(400), 10), dec=rng.integers(0, 2, 4000),
                                  match=rng.integers(0, 2, 4000), dec_o=rng.integers(0, 2, 4000),
                                  noise=rng.random(4000), tiebreak=rng.random(4000)))
        self.assertAlmostEqual(em.top_k(frame, "noise", 3)["dec"].mean(), frame["dec"].mean(), delta=0.03)

    def test_shuffled_order_gives_the_mechanical_lag_and_no_contrast(self):
        """Without real sequence, previous attractiveness is about 0 and the
        lagged yes sits near -1/(T-1). This is why estimates need a placebo."""
        frame = ef.add_consensus(self.data).dropna(subset=[*ef.CONSENSUS, "order", "dec"])
        rng = np.random.default_rng(5)
        draws = np.array([em.contrast_estimates(frame, rng) for _ in range(40)])
        self.assertLess(abs(draws[:, 0].mean()), 0.5)
        self.assertLess(draws[:, 2].mean(), -4)

    def test_gini_of_equal_exposure_is_zero(self):
        self.assertAlmostEqual(em.gini([3, 3, 3, 3]), 0.0)


class KaggleAdaptations(Shared):
    """A5j endings, A5k direct match model and S4 common ground."""

    def test_endings_sum_to_one_and_keep_the_match_share(self):
        rng = np.random.default_rng(6)
        dec, dec_o = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
        frame = pd.DataFrame(dict(dec=dec, dec_o=dec_o, match=dec * dec_o)).astype(float)
        shares = em.endings(frame)
        np.testing.assert_allclose(shares.sum(axis=1), 1.0)
        self.assertAlmostEqual(shares["match"].mean(), frame["match"].mean())
        self.assertTrue((shares >= 0).all().all())

    def test_unreturned_yes_is_counted_as_only_i(self):
        one = pd.DataFrame(dict(dec=[1.0], dec_o=[0.0], match=[0.0]))
        self.assertEqual(em.endings(one).iloc[0].to_dict(),
                         dict(neither=0.0, only_i_said_yes=1.0, only_they_said_yes=0.0, match=0.0))

    def test_direct_model_uses_no_outcome_of_the_row(self):
        frame = ef.add_pair_features(ef.add_split_fold(self.data)).dropna(subset=ef.SPLIT_FOLD).head(50)
        x = em.match_features(frame)
        self.assertFalse(set(x.columns) & {"dec", "dec_o", "match"})
        # Columns named after a trait hold the partner's STATED weight, never
        # this row's post-date rating.
        for attribute in ATTRS:
            expected = frame[ef.PARTNER_STATED[attribute]].fillna(16.7).fillna(0) / 100
            np.testing.assert_allclose(x[attribute], expected.fillna(0))

    def test_adding_the_direct_model_leaves_published_rankings_alone(self):
        ctx = es.Context(self.data, self.sample, 10)
        scored = em.ranking_frame(ctx)
        self.assertFalse(scored["direct_match"].isna().any())
        saved = pd.read_csv(project_root() / TABLES_DIR / "ext_a5_ranking.csv")
        saved = saved[saved["top_k"] == 3].set_index("rule")["match_rate"]
        for rule in ("stated", "revealed", "reciprocal"):
            self.assertAlmostEqual(em.top_k(scored, rule, 3)["match"].mean(),
                                   saved[em.RULE_LABELS[rule]], places=5)

    def test_common_ground_bands_cover_every_row(self):
        result = em.s4_common_ground(es.Context(self.data, self.sample, 20))
        bands = result["tables"]["s4_common_ground"]
        frame = self.sample.assign(agegap=(self.sample["age"] - self.sample["age_o"]).abs())
        for column, label, _, _ in em.COMMON_GROUND:
            self.assertEqual(bands.loc[bands["measure"] == label, "dates"].sum(),
                             frame[column].notna().sum())
        self.assertTrue(np.isfinite(result["tables"]["s4_common_ground_slopes"]["pp_per_sd"]).all())


class MirrorModel(Shared):
    """A5l: model B as the mirror of model A."""

    def test_your_ratings_are_constant_within_your_list(self):
        """If they varied by partner they could steer the ranking (module rule)."""
        frame = ef.add_reverse_ratings(self.data)
        self.assertEqual(frame.groupby("iid")["you_attr"].nunique().max(), 1)

    def test_personal_weights_leave_out_the_partner(self):
        """Flipping j's decision about i must not move j's weights used for i."""
        rows = ef.add_split_fold(self.data)
        rows = rows[(rows["gender"] == 1) & (rows["wave"] == 2)].dropna(subset=ef.SPLIT_FOLD)
        prior, mean, sd = np.zeros(6), rows[ef.SPLIT_FOLD].mean().to_numpy(), \
            rows[ef.SPLIT_FOLD].std().to_numpy()
        first = ef.personal_weights(rows, prior, mean, sd)
        altered = rows.copy()
        target = altered.index[0]
        altered.loc[target, "dec"] = 1 - altered.loc[target, "dec"]
        second = ef.personal_weights(altered, prior, mean, sd)
        key = (rows.loc[target, "iid"], rows.loc[target, "pid"])
        pick = lambda w: w[(w["iid"] == key[0]) & (w["pid"] == key[1])][ef.PERSONAL].to_numpy()
        np.testing.assert_allclose(pick(first), pick(second))
        other = (first["iid"] == key[0]) & (first["pid"] != key[1])
        self.assertFalse(np.allclose(first.loc[other, ef.PERSONAL], second.loc[other, ef.PERSONAL]))

    def test_partner_weights_from_n_dates_leave_out_the_partner(self):
        """A5n: with n of the partner's dates, the date with you is still excluded
        and at most n dates are used."""
        rows = ef.add_split_fold(self.data)
        rows = rows[(rows["gender"] == 1) & (rows["wave"] == 2)].dropna(subset=ef.SPLIT_FOLD)
        prior, mean, sd = np.zeros(6), rows[ef.SPLIT_FOLD].mean().to_numpy(), \
            rows[ef.SPLIT_FOLD].std().to_numpy()
        first = ef.personal_weights(rows, prior, mean, sd, n=4, seed=1)
        altered = rows.copy()
        target = altered.index[0]
        altered.loc[target, "dec"] = 1 - altered.loc[target, "dec"]
        second = ef.personal_weights(altered, prior, mean, sd, n=4, seed=1)
        key = (rows.loc[target, "iid"], rows.loc[target, "pid"])
        pick = lambda w: w[(w["iid"] == key[0]) & (w["pid"] == key[1])][ef.PERSONAL].to_numpy()
        np.testing.assert_allclose(pick(first), pick(second))
        smaller = ef.personal_weights(rows, prior, mean, sd, n=2, seed=1)
        self.assertFalse(np.allclose(first[ef.PERSONAL], smaller[ef.PERSONAL]))

    def test_shrinkage_pulls_to_the_prior_and_releases_to_the_data(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(4000, 6))
        truth = np.array([1.5, -0.5, 0.0, 0.8, 0.0, 0.3])
        y = (rng.random(4000) < 1 / (1 + np.exp(-(x @ truth)))).astype(float)
        prior = np.full(6, 0.2)
        np.testing.assert_allclose(ef._shrunk_logit(x, y, prior, 1e7), prior, atol=1e-3)
        np.testing.assert_allclose(ef._shrunk_logit(x, y, prior, 1e-6), truth, atol=0.15)

    def test_stated_fit_is_their_weights_times_your_ratings(self):
        scored = em.ranking_frame(es.Context(self.data, self.sample, 10), mode="mirror_stated")
        row = scored.iloc[0]
        ratings = self.data[self.data["pid"] == row["iid"]][list(ATTRS)].mean()
        expected = sum(np.nan_to_num(row[ef.PARTNER_STATED[a]], nan=16.7) / 100 * ratings[a]
                       for a in ATTRS)
        self.assertAlmostEqual(row["fit_stated"], expected, places=6)


class LearningCurve(unittest.TestCase):
    """A5m: weights learned from n dates, scored on a held-out date."""

    def setUp(self):
        rng = np.random.default_rng(9)
        self.x = np.ascontiguousarray(rng.normal(size=(16, 6)))
        self.y = (rng.random(16) < 0.5).astype(float)
        self.prior = np.full(6, 0.3)

    def test_held_out_decisions_never_train_the_model(self):
        """Flip every held-out label: if the model never saw them, each AUC
        becomes exactly 1 - AUC (same scores, reversed labels)."""
        stated = self.x[:, 0].copy()
        first = ef.learning_draws(self.x, self.y, stated, self.prior, 1.0, 8, 4, 1, 11)
        np.random.seed(11)
        test = np.random.permutation(16)[:4]
        flipped = self.y.copy()
        flipped[test] = 1 - flipped[test]
        second = ef.learning_draws(self.x, flipped, stated, self.prior, 1.0, 8, 4, 1, 11)
        # Flipping every held-out label reverses each AUC exactly (1 - AUC).
        np.testing.assert_allclose(second[0], 1 - first[0])

    def test_heavy_shrinkage_matches_the_pooled_weights(self):
        stated = self.x[:, 0].copy()
        out = ef.learning_draws(self.x, self.y, stated, self.prior, 1e7, 5, 4, 20, 3)
        both = ~np.isnan(out[:, 0])
        np.testing.assert_allclose(out[both, 0], out[both, 2])

    def test_held_out_dates_are_the_same_for_every_n(self):
        stated = self.x[:, 0].copy()
        small = ef.learning_draws(self.x, self.y, stated, self.prior, 1.0, 2, 4, 10, 5)
        large = ef.learning_draws(self.x, self.y, stated, self.prior, 1.0, 10, 4, 10, 5)
        np.testing.assert_allclose(small[:, 1:], large[:, 1:])


class LatentClass(Shared):
    def test_one_class_equals_the_conditional_logit(self):
        strata = ef.make_strata(self.sample[self.sample["gender"] == 0], ATTRS)
        fit = ef.fit_clogit(strata)
        x = np.ascontiguousarray(strata.centred / fit.sd)
        total, betas, _ = es.latent_class(strata, x, 1, 0)
        self.assertAlmostEqual(total, fit.loglike, places=2)
        np.testing.assert_allclose(betas[0], fit.beta, atol=1e-2)


class Inference(unittest.TestCase):
    def test_bh_matches_hand_calculation(self):
        np.testing.assert_allclose(ef.bh_adjust([0.01, 0.04, 0.03, 0.2]),
                                   [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)

    def test_relative_weights_sum_to_r_squared(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(500, 4))
        x[:, 1] += x[:, 0]
        y = x @ np.array([1.0, 0.5, 0.0, -0.3]) + rng.normal(size=500)
        z = np.c_[np.ones(500), x]
        fitted = z @ np.linalg.lstsq(z, y, rcond=None)[0]
        r2 = 1 - ((y - fitted)**2).sum() / ((y - y.mean())**2).sum()
        self.assertAlmostEqual(ef.relative_weights(x, y).sum(), r2, places=6)

    def test_bootstrap_p_is_large_for_noise_around_zero(self):
        values = np.random.default_rng(2).normal(size=200)
        self.assertGreater(ef.boot_mean(values - values.mean())["p"], 0.5)


if __name__ == "__main__":
    unittest.main()
