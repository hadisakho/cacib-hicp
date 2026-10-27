"""Contrôles des dates d'apprentissage et de la prévision du boosting."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from boosting_model import BOOST_SPEC, boosting_forecast
from ridge_model import inner_splits, make_design


class BoostingTests(unittest.TestCase):
    def series(self, length=150):
        t = np.arange(length)
        return pd.Series(
            80 * np.exp(0.0015 * t + 0.008 * np.sin(2 * np.pi * t / 12)),
            index=pd.period_range("2000-01", periods=length, freq="M"),
        )

    def test_constant_growth_target_and_index_conversion(self):
        series = self.series()
        series[:] = 80 * 1.002 ** np.arange(len(series))
        result = boosting_forecast(series)
        np.testing.assert_allclose(
            result, series.iloc[-1] * 1.002 ** np.arange(1, 4), rtol=1e-7
        )

    def test_reproducibility_and_past_only_labels(self):
        series = self.series()
        first = boosting_forecast(series)
        pd.testing.assert_series_equal(first, boosting_forecast(series))
        self.assertTrue(np.isfinite(first).all() and first.gt(0).all())
        for details in first.attrs["fit_details"]["boosting_by_horizon"].values():
            self.assertLessEqual(details["last_training_target"], str(series.index[-1]))
            self.assertIn(
                details["max_leaf_nodes"], BOOST_SPEC["candidate_max_leaf_nodes"]
            )
            for fold in details["folds"]:
                self.assertLessEqual(
                    fold["train_last_label"], fold["validation_first_origin"]
                )
                self.assertLessEqual(
                    fold["validation_last_label"], str(series.index[-1])
                )

    def test_actual_fit_receives_only_training_fold(self):
        series = self.series()
        captured = []

        class RecordingBoost(HistGradientBoostingRegressor):
            def fit(self, X, y, **kwargs):
                captured.append(X.copy())
                return super().fit(X, y, **kwargs)

        with patch("boosting_model.HistGradientBoostingRegressor", RecordingBoost):
            boosting_forecast(series, 1)
        x, _, _ = make_design(series, 1)
        train, _ = inner_splits(x, 1)[0]
        pd.testing.assert_frame_equal(captured[0], x.iloc[train])
        pd.testing.assert_frame_equal(captured[-1], x)

    def test_short_history_is_rejected(self):
        with self.assertRaises(ValueError):
            boosting_forecast(self.series(60))


if __name__ == "__main__":
    unittest.main()
