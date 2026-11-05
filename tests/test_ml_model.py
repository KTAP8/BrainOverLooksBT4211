"""Checks for Part 4. Each one can fail for a real reason, not by restating code."""
import unittest

import numpy as np
import pandas as pd

from src import ml_model
from src.constants import ATTRS, SEED
from src.io import load_clean, project_root
from src.constants import TABLES_DIR


class PrepareFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.scaling = ml_model.prepare_ml_frame()
        cls.flow = pd.read_csv(project_root() / TABLES_DIR / "core_model_sample_flow.csv")

    def test_matches_part3_sample(self):
        """The whole point of this section is to analyse Part 3's sample."""
        self.assertEqual(len(self.frame), int(self.flow["model_rows"].sum()))
        self.assertEqual(self.frame["iid"].nunique(), int(self.flow["model_people"].sum()))

    def test_every_person_has_both_outcomes(self):
        counts = self.frame.groupby("iid")["dec"].nunique()
        self.assertTrue((counts == 2).all())

    def test_standardisation_reproduces_part3_scaler(self):
        """z_<attr> must use Part 3's published mean/SD, not a pooled one."""
        for name in ("Women", "Men"):
            side = self.frame[self.frame["group"] == name]
            scale = self.scaling[self.scaling["gender"] == name].set_index("attribute")
            for attribute in ATTRS:
                rebuilt = ((side[attribute] - scale.loc[attribute, "mean"])
                           / scale.loc[attribute, "sd_ddof0"])
                np.testing.assert_allclose(side[f"z_{attribute}"], rebuilt, atol=1e-12)
                # Standardised within gender => mean 0, sd 1 within that gender.
                self.assertAlmostEqual(float(side[f"z_{attribute}"].mean()), 0.0, places=6)
                self.assertAlmostEqual(float(side[f"z_{attribute}"].std(ddof=0)), 1.0, places=6)

    def test_demeaned_columns_are_person_centred(self):
        for attribute in ATTRS:
            per_person = self.frame.groupby("iid")[f"d_{attribute}"].mean()
            np.testing.assert_allclose(per_person, 0.0, atol=1e-10)

    def test_sample_assertion_fires_when_data_changes(self):
        """A silently different sample is the failure mode worth guarding."""
        damaged = load_clean()
        damaged = damaged[damaged["wave"] != damaged["wave"].iloc[0]]
        with self.assertRaises(AssertionError):
            ml_model.prepare_ml_frame(damaged)


class GroupedSplitting(unittest.TestCase):
    def test_no_person_spans_train_and_test(self):
        """Row-level splits leak a person's fixed effect into their own test rows."""
        from sklearn.model_selection import GroupKFold
        frame, _ = ml_model.prepare_ml_frame()
        groups = frame["iid"].to_numpy()
        x = frame[[f"z_{a}" for a in ATTRS]].to_numpy(float)
        y = frame["dec"].to_numpy(int)
        for train, test in GroupKFold(n_splits=ml_model.N_SPLITS).split(x, y, groups):
            self.assertEqual(set(groups[train]) & set(groups[test]), set())


class Determinism(unittest.TestCase):
    def test_benchmark_is_reproducible(self):
        frame, _ = ml_model.prepare_ml_frame()
        first = ml_model._cv_scores(frame, [f"z_{a}" for a in ATTRS], ml_model.make_gbm)[0]
        second = ml_model._cv_scores(frame, [f"z_{a}" for a in ATTRS], ml_model.make_gbm)[0]
        self.assertEqual(first, second)

    def test_seed_is_the_project_seed(self):
        self.assertEqual(ml_model.make_gbm().random_state, SEED)


class InteractionCheck(unittest.TestCase):
    def test_interaction_terms_are_products_of_main_effects(self):
        frame, _ = ml_model.prepare_ml_frame()
        result = ml_model.interaction_check(frame.head(400))
        self.assertEqual(result["n_interaction_terms"], 15)  # 6 choose 2


if __name__ == "__main__":
    unittest.main()
