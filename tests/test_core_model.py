"""Numerical checks for the exact likelihood and bootstrap implementation.

Run from root: python -m unittest discover -s tests -v
"""
import itertools
import unittest
import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from src.constants import ATTRS, STATED
from src.core_model import conditional_objective, prepare_sample, fit_sample, revealed_weights


class CoreModelTests(unittest.TestCase):
    def test_exact_likelihood_and_gradient(self):
        x = np.array([[1., 2.], [2., -1.], [-1., 1.], [0., 0.]])
        y = np.array([1., 0., 1., 0.])
        beta = np.array([.2, -.1])
        args = (x, y, np.array([0, 4]), np.array([2]), np.ones(1))
        value, gradient = conditional_objective(beta, *args)
        den = sum(np.exp((x[list(c)] @ beta).sum()) for c in itertools.combinations(range(4), 2))
        self.assertAlmostEqual(value, np.log(den)-y @ (x @ beta), places=10)
        numerical = np.array([(conditional_objective(beta+np.eye(2)[k]*1e-5, *args)[0]
                              - conditional_objective(beta-np.eye(2)[k]*1e-5, *args)[0])/2e-5
                              for k in range(2)])
        np.testing.assert_allclose(gradient, numerical, atol=1e-7)

    def test_person_bootstrap_matches_explicit_new_strata(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.uniform(1, 10, (120, 6)), columns=ATTRS)
        data["iid"] = np.repeat(np.arange(15), 8)
        data["gender"] = 0
        data["wave"] = 1
        data["dec"] = np.tile([0, 1, 0, 1, 0, 1, 0, 1], 15)
        data[STATED] = 100/6
        # Include two noncontributing people to test sample exclusions.
        data.loc[data.iid == 0, "dec"] = 0
        data.loc[data.iid == 1, "dec"] = 1
        sample, flow = prepare_sample(data)
        self.assertEqual(flow["excluded_constant_choice_people"], 2)
        self.assertEqual(flow["model_people"], 13)
        ids = sample.frame.iid.unique()
        selected = rng.choice(np.arange(len(ids)), len(ids), replace=True)
        mult = np.bincount(selected, minlength=len(ids)).astype(float)
        fitted = fit_sample(sample, mult)
        blocks = []
        for new_id, index in enumerate(selected):
            block = sample.frame[sample.frame.iid == ids[index]].copy()
            block["bootstrap_iid"] = new_id
            blocks.append(block)
        duplicate = pd.concat(blocks, ignore_index=True)
        x = duplicate[ATTRS].to_numpy()
        z = (x-x.mean(axis=0))/x.std(axis=0, ddof=0)
        reference = ConditionalLogit(duplicate.dec, z, groups=duplicate.bootstrap_iid).fit(
            method="bfgs", disp=False, maxiter=300)
        np.testing.assert_allclose(fitted["beta"], reference.params, atol=0.001)
        np.testing.assert_allclose(fitted["sd"], x.std(axis=0), atol=1e-10)

    def test_clipping_does_not_create_negative_importance(self):
        np.testing.assert_allclose(revealed_weights(np.array([2., -1., 0., 2.])), [50, 0, 0, 50])
        with self.assertRaises(ValueError):
            revealed_weights(np.array([-1., 0.]))


if __name__ == "__main__":
    unittest.main()
