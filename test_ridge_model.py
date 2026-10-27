"""Vérifie les dates d'information, le calcul des cibles et le scaler interne."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ridge_model import RIDGE_SPEC, inner_splits, make_design, ridge_forecast


class RidgeTests(unittest.TestCase):
    def series(self, length=180):
        t = np.arange(length)
        values = 80 * np.exp(0.0015 * t + 0.008 * np.sin(2 * np.pi * t / 12))
        return pd.Series(
            values, index=pd.period_range("2000-01", periods=length, freq="M")
        )

    def test_labels_and_known_base_are_correct(self):
        series = self.series()
        for h in (1, 2, 3):
            x, y, query = make_design(series, h)
            origin = x.index[30]
            self.assertAlmostEqual(
                y.loc[origin],
                100 * (series.loc[origin + h] / series.loc[origin + h - 12] - 1),
            )
            self.assertAlmostEqual(
                x.loc[origin, "known_annual_part"],
                100 * (series.loc[origin] / series.loc[origin + h - 12] - 1),
            )
            self.assertLessEqual(x.index[-1] + h, series.index[-1])
            self.assertEqual(query.index[0], series.index[-1])

    def test_features_do_not_use_future_indices(self):
        full = self.series()
        past = full.iloc[:140]
        for h in (1, 2, 3):
            _, _, query = make_design(past, h)
            full.iloc[140:] *= 3
            x, _, _ = make_design(full, h)
            pd.testing.assert_frame_equal(query, x.loc[query.index])

    def test_inner_training_labels_are_available_at_validation_origin(self):
        for h in (1, 2, 3):
            x, _, _ = make_design(self.series(), h)
            for train, validation in inner_splits(x, h):
                self.assertLessEqual(x.index[train[-1]] + h, x.index[validation[0]])
                self.assertEqual(len(validation), 12)

    def test_scaler_fits_only_inner_training_rows(self):
        series = self.series()
        x, _, _ = make_design(series, 1)
        first_train, _ = inner_splits(x, 1)[0]
        captured = []

        class RecordingScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                captured.append(X.copy())
                return super().fit(X, y, sample_weight=sample_weight)

        with patch("ridge_model.StandardScaler", RecordingScaler):
            ridge_forecast(series, 1)
        pd.testing.assert_frame_equal(captured[0], x.iloc[first_train])
        pd.testing.assert_frame_equal(captured[-1], x)

    def test_constant_monthly_growth_and_rebasing(self):
        series = self.series()
        series[:] = 80 * 1.002 ** np.arange(len(series))
        prediction = ridge_forecast(series)
        expected = series.iloc[-1] * 1.002 ** np.arange(1, 4)
        np.testing.assert_allclose(prediction, expected, rtol=1e-8)
        scaled = ridge_forecast(series * 7)
        np.testing.assert_allclose(scaled, prediction * 7, rtol=1e-8)
        for info in prediction.attrs["fit_details"]["ridge_by_horizon"].values():
            self.assertIn(info["alpha"], RIDGE_SPEC["alphas"])

    def test_short_history_is_rejected(self):
        with self.assertRaises(ValueError):
            ridge_forecast(self.series(60))


if __name__ == "__main__":
    unittest.main()
