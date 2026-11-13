"""Checks for A5o (src/ext_same_model.py). Each one can fail for a real reason."""
import unittest

import numpy as np
import pandas as pd

from src import ext_same_model as sm


class LeaveOneOutRate(unittest.TestCase):
    def test_partner_rate_excludes_their_decision_about_you(self):
        # One chooser, four partners: yes, yes, no, no. The rate on each row must be
        # the mean of the other three decisions, so flipping a row's own decision
        # leaves that row's rate unchanged.
        frame = pd.DataFrame({"iid": [1, 1, 1, 1], "dec": [1, 1, 0, 0]})
        rate = sm.leave_one_out_rate(frame)
        np.testing.assert_allclose(rate, [1 / 3, 1 / 3, 2 / 3, 2 / 3])
        flipped = frame.assign(dec=[0, 1, 0, 0])
        self.assertAlmostEqual(sm.leave_one_out_rate(flipped).iloc[0], rate.iloc[0])

    def test_rate_is_clipped(self):
        frame = pd.DataFrame({"iid": [1, 1, 1], "dec": [1, 1, 1]})
        self.assertTrue((sm.leave_one_out_rate(frame) <= 0.98).all())


class OwnSide(unittest.TestCase):
    def test_centring_keeps_list_mean_at_the_yes_rate_logit(self):
        # Utility centred within each list: at the list's average partner the
        # probability equals the yes rate, so the rate only sets the level.
        frame = pd.DataFrame({"iid": [1, 1, 2, 2], **{c: [0.0, 1.0, 0.0, 1.0] for c in sm.SF + sm.PAIR}})
        params = np.ones(len(sm.SF + sm.PAIR))
        rate = pd.Series([0.2, 0.2, 0.8, 0.8])
        prob = sm.own_side(frame, params, rate)
        logit = np.log(prob / (1 - prob))
        np.testing.assert_allclose(logit.groupby(frame["iid"]).mean(),
                                   np.log(np.array([0.2, 0.8]) / np.array([0.8, 0.2])))


class PublishedTable(unittest.TestCase):
    def test_b0_reproduces_a5_and_reverse_rows_exist(self):
        table = pd.read_csv(sm.OUT)
        check = pd.read_csv(sm.ROOT / "outputs" / "tables" / "ext_a5_seed_sensitivity.csv").set_index("seed")
        b0 = table[table["rule"] == sm.RULES["b0_mixed"]].set_index("seed")["top3_match"]
        np.testing.assert_allclose(b0, check.loc[b0.index, "reciprocal_match"], atol=1e-6)
        self.assertLess(table["reverse_rows_missing"].max(), 0.01)


if __name__ == "__main__":
    unittest.main()
