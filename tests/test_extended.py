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
