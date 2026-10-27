"""Vérifie le décalage calendaire et la conservation des valeurs manquantes."""

import unittest

import numpy as np
import pandas as pd

from prepare_external import SECTORS, make_features


class ExternalTests(unittest.TestCase):
    def indices(self):
        months = pd.period_range("2000-01", periods=48, freq="M")
        return pd.DataFrame(
            {name: 100 * 1.01 ** np.arange(48) for name in SECTORS}, index=months
        )

    def test_reference_month_and_percentages(self):
        data = self.indices()
        result = make_features(data, data.index)
        self.assertAlmostEqual(result.loc["2001-03", "ppi_energy_mom"], 1.0)
        self.assertAlmostEqual(
            result.loc["2001-03", "ppi_energy_yoy"], 100 * (1.01**12 - 1)
        )
        self.assertTrue(result.loc[:"2001-02", "ppi_energy_yoy"].isna().all())

    def test_future_observations_cannot_change_earlier_features(self):
        data = self.indices()
        original = make_features(data, data.index)
        data.loc["2002-01":] *= 3
        changed = make_features(data, data.index)
        pd.testing.assert_frame_equal(original.loc[:"2002-02"], changed.loc[:"2002-02"])

    def test_missing_month_is_not_filled_or_shifted_by_row(self):
        data = self.indices().drop(pd.Period("2001-01", freq="M"))
        origins = pd.period_range("2000-01", "2003-12", freq="M")
        result = make_features(data, origins)
        self.assertTrue(result.loc["2001-03"].isna().all())
        self.assertTrue(pd.isna(result.loc["2001-04", "ppi_energy_mom"]))
        self.assertAlmostEqual(result.loc["2001-05", "ppi_energy_mom"], 1.0)

    def test_negative_indices_and_short_lag_are_rejected(self):
        data = self.indices()
        with self.assertRaises(ValueError):
            make_features(data, data.index, lag_months=0)
        data.iloc[10, 0] = -1
        with self.assertRaises(ValueError):
            make_features(data, data.index)


if __name__ == "__main__":
    unittest.main()
