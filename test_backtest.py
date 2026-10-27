"""Tests portables du protocole : aucune connexion à Eurostat nécessaire."""

import unittest
import warnings

import numpy as np
import pandas as pd

from backtest import rolling_backtest, summarize, validation_bounds
from baselines import baseline_forecast, level_to_yoy, validate_series
from forecast_models import forecast_levels


def synthetic_history() -> pd.Series:
    index = pd.period_range("2000-01", "2026-09", freq="M")
    n = np.arange(len(index))
    values = 100.0 * np.exp(0.002 * n + 0.01 * np.sin(2 * np.pi * n / 12))
    return pd.Series(values, index=index)


class ForecastTests(unittest.TestCase):
    def test_flat_yoy_keeps_annual_rate_at_all_three_horizons(self):
        train = synthetic_history().loc[:"2020-09"]
        prediction = baseline_forecast(train, "flat_yoy")
        current = 100 * (train.iloc[-1] / train.loc[train.index[-1] - 12] - 1)
        for target, level in prediction.items():
            self.assertAlmostEqual(
                level_to_yoy(train, level, target), current, places=10
            )

    def test_flat_index_only_changes_yoy_through_known_base(self):
        train = synthetic_history().loc[:"2020-09"]
        prediction = baseline_forecast(train, "flat_index")
        self.assertTrue(prediction.eq(train.iloc[-1]).all())
        target = train.index[-1] + 3
        expected = 100 * (train.iloc[-1] / train.loc[target - 12] - 1)
        self.assertAlmostEqual(
            level_to_yoy(train, prediction.loc[target], target), expected
        )

    def test_seasonal_reference_recovers_repeated_relative_paths(self):
        full = synthetic_history()
        train = full.loc[:"2020-09"]
        prediction = baseline_forecast(train, "seasonal_mean")
        np.testing.assert_allclose(prediction, full.loc[prediction.index], rtol=1e-12)

    def test_yoy_is_not_annualized_three_month_inflation(self):
        full = synthetic_history()
        origin = pd.Period("2020-08", freq="M")
        train = full.loc[:origin]
        target = origin + 3
        level = float(train.iloc[-1]) * 1.03
        yoy = level_to_yoy(train, level, target)
        self.assertAlmostEqual(yoy, 100 * (level / train.loc[target - 12] - 1))
        self.assertNotAlmostEqual(yoy, 400 * np.log(level / train.iloc[-1]), places=3)
        with self.assertRaises(ValueError):
            level_to_yoy(train, level, origin + 13)

    def test_invalid_or_incomplete_training_series_is_rejected(self):
        base = synthetic_history().iloc[:30]
        broken = [
            base.drop(base.index[4]),
            base.iloc[::-1],
            pd.concat([base, base.iloc[-1:]]),
        ]
        for value in (0, -1, np.nan, np.inf):
            series = base.copy()
            series.iloc[5] = value
            broken.append(series)
        for series in broken:
            with self.subTest(index=len(series)), self.assertRaises(ValueError):
                validate_series(series)

    def test_reserved_final_test_dates_match_the_agreed_protocol(self):
        start, end, test_start, test_end = validation_bounds("2026-09")
        self.assertEqual(str(start), "2014-01")
        self.assertEqual(str(end), "2024-09")
        self.assertEqual(str(test_start), "2024-10")
        self.assertEqual(str(test_end), "2026-09")

    def test_future_perturbation_cannot_change_an_origin_prediction(self):
        history = synthetic_history()
        changed = history.copy()
        origin = pd.Period("2018-03", freq="M")
        changed.loc[origin + 1 :] *= 1.7
        models = ["flat_index", "flat_yoy", "seasonal_mean"]
        first = rolling_backtest(history, models, "2018-04", "2018-06")
        second = rolling_backtest(changed, models, "2018-04", "2018-06")
        left = first.loc[first["origin"].eq(str(origin)), "predicted_index"]
        right = second.loc[second["origin"].eq(str(origin)), "predicted_index"]
        np.testing.assert_array_equal(left, right)

    def test_rolling_window_matrix_and_last_training_month(self):
        predictions = rolling_backtest(
            synthetic_history(),
            ["flat_index", "flat_yoy"],
            "2018-01",
            "2018-06",
            window_months=120,
        )
        self.assertEqual(len(predictions), 2 * 3 * 6)
        self.assertTrue(predictions["train_months"].eq(120).all())
        self.assertTrue(predictions["train_end"].eq(predictions["origin"]).all())
        for row in predictions.itertuples():
            self.assertEqual(
                pd.Period(row.target, freq="M") - row.horizon,
                pd.Period(row.origin, freq="M"),
            )
            self.assertLessEqual(
                pd.Period(row.base_month, freq="M"), pd.Period(row.origin, freq="M")
            )

    def test_failed_fit_does_not_improve_score_by_dropping_dates(self):
        def predictor(train, model, horizon):
            if model == "broken" and train.index[-1] == pd.Period("2017-12", freq="M"):
                raise RuntimeError("fixture convergence error")
            return baseline_forecast(train, "flat_index", horizon)

        predictions = rolling_backtest(
            synthetic_history(),
            ["flat_index", "broken"],
            "2018-01",
            "2018-06",
            forecaster=predictor,
        )
        scores = summarize(predictions)
        broken = scores.loc[scores["model"].eq("broken")]
        self.assertFalse(broken["eligible"].any())
        self.assertTrue(broken["rmse_pp"].isna().all())
        self.assertTrue(scores.loc[scores["model"].eq("flat_index"), "eligible"].all())
        self.assertEqual(len(predictions), 2 * 3 * 6)

    def test_scores_are_in_percentage_points(self):
        predictions = pd.DataFrame(
            {
                "model": ["fixture"] * 2,
                "horizon": [3, 3],
                "status": ["ok", "ok"],
                "error_pp": [-1.0, 3.0],
            }
        )
        scores = summarize(predictions).iloc[0]
        self.assertEqual(scores.mae_pp, 2.0)
        self.assertAlmostEqual(scores.rmse_pp, np.sqrt(5))
        self.assertEqual(scores.bias_pp, 1.0)

    def test_fitted_models_return_finite_positive_monthly_forecasts(self):
        rng = np.random.default_rng(42)
        train = synthetic_history().iloc[:180].copy()
        train *= np.exp(np.cumsum(rng.normal(0, 0.001, len(train))))
        for model in ("sarima_airline", "ets_additive"):
            with self.subTest(model=model), warnings.catch_warnings(record=True):
                prediction = forecast_levels(train, model)
                self.assertEqual(len(prediction), 3)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(prediction.gt(0).all())
                self.assertTrue(prediction.attrs["fit_details"]["converged"])


if __name__ == "__main__":
    unittest.main()
