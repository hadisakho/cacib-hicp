"""Tests indépendants : poids variables, chaînage et données manquantes."""

import unittest

import numpy as np
import pandas as pd
from check_aggregation import BRANCHES, reconstruct


class AggregationTests(unittest.TestCase):
    def fixture(self):
        months = pd.period_range("2000-12", "2002-12", freq="M")
        levels = pd.DataFrame(100.0, index=months, columns=["TOTAL", *BRANCHES])
        # Une branche augmente de 10 % par an, puis 20 % la seconde année.
        levels.loc["2001-01":"2001-12", BRANCHES[0]] = 110.0
        levels.loc["2002-01":"2002-12", BRANCHES[0]] = 132.0
        weights = pd.DataFrame(200.0, index=[2001, 2002], columns=BRANCHES)
        weights.loc[2002] = [400, 150, 150, 150, 150]
        levels.loc["2001-01":"2001-12", "TOTAL"] = 102.0
        levels.loc["2002-01":"2002-12", "TOTAL"] = 110.16
        return levels, weights

    def test_changing_weights_and_december_chain(self):
        levels, weights = self.fixture()
        result = reconstruct(levels, weights, "2001-01", "2002-12")
        np.testing.assert_allclose(result["chain_reconstructed"], result["official"])
        self.assertAlmostEqual(result.loc["2002-01", "chain_reconstructed_yoy"], 8.0)

    def test_continuous_chain_does_not_reset_to_official(self):
        levels, weights = self.fixture()
        levels.loc["2001-12", "TOTAL"] = 150.0
        result = reconstruct(levels, weights, "2001-01", "2002-12")
        self.assertAlmostEqual(result.loc["2002-01", "chain_reconstructed"], 110.16)
        self.assertAlmostEqual(
            result.loc["2002-01", "annual_anchor_reconstructed"], 162.0
        )

    def test_missing_weight_is_not_silently_ignored(self):
        levels, weights = self.fixture()
        weights.loc[2001, BRANCHES[0]] = np.nan
        with self.assertRaises(ValueError):
            reconstruct(levels, weights, "2001-01", "2002-12")


if __name__ == "__main__":
    unittest.main()
