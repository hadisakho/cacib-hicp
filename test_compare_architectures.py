"""Tests du franchissement d'année et de l'information passée utilisée."""

import unittest

import numpy as np
import pandas as pd

from check_aggregation import BRANCHES
from compare_architectures import aggregate_paths, make_predictor


class ArchitectureTests(unittest.TestCase):
    def fixture(self):
        dates = pd.period_range("2000-12", "2001-11", freq="M")
        panel = pd.DataFrame(100.0, index=dates, columns=["TOTAL", *BRANCHES])
        future = pd.DataFrame(
            100.0,
            index=pd.period_range("2001-12", periods=3, freq="M"),
            columns=BRANCHES,
        )
        future.loc["2001-12", BRANCHES[0]] = 110
        future.loc["2002-01":, BRANCHES[0]] = 121
        weights = pd.DataFrame(
            [[200] * 5, [400, 150, 150, 150, 150]], index=[2001, 2002], columns=BRANCHES
        )
        return panel, future, weights

    def test_january_uses_predicted_december_and_old_weights(self):
        panel, future, weights = self.fixture()
        result = aggregate_paths(panel, future, weights)
        self.assertAlmostEqual(result.loc["2001-12"], 102.0)
        self.assertAlmostEqual(result.loc["2002-01"], 104.04)
        self.assertEqual(result.attrs["fit_details"]["weights_year"], 2001)

    def test_future_weights_cannot_change_forecast(self):
        panel, future, weights = self.fixture()
        original = aggregate_paths(panel, future, weights)
        weights.loc[2002] = [1000, 0, 0, 0, 0]
        pd.testing.assert_series_equal(
            original, aggregate_paths(panel, future, weights)
        )

    def test_missing_origin_weights_are_rejected(self):
        panel, future, weights = self.fixture()
        with self.assertRaises(ValueError):
            aggregate_paths(panel, future, weights.loc[[2002]])

    def test_callback_never_passes_future_indices_to_models(self):
        panel, future, weights = self.fixture()
        full = pd.concat([panel, future.assign(TOTAL=999.0)])
        received = []

        def fake(train, model, horizon):
            received.append(train.copy())
            self.assertEqual(train.index[-1], panel.index[-1])
            return pd.Series(100.0, index=future.index)

        predictor = make_predictor(full, weights, model_predictor=fake)
        predictor(panel["TOTAL"], "bottom_up__flat_yoy", 3)
        self.assertEqual(len(received), 5)
        self.assertTrue(all(len(s) == 12 for s in received))

    def test_flat_components_reproduce_flat_headline(self):
        panel, future, weights = self.fixture()
        future.loc[:] = 100
        np.testing.assert_allclose(aggregate_paths(panel, future, weights), 100)


if __name__ == "__main__":
    unittest.main()
