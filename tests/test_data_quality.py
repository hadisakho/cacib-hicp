import numpy as np
import pandas as pd
import pytest

from hicp.config import load_config
from hicp.quality import DataQualityError, clean_indices, clean_weights, series_report
from hicp.synthetic import make_demo_data
from hicp.validation import run_checks


def _frame(rows):
    df = pd.DataFrame(rows, columns=["time", "geo", "coicop", "value", "flag"])
    df["time"] = pd.PeriodIndex(df["time"], freq="M")
    return df


def test_exact_duplicates_dropped_and_bad_values_removed():
    df = _frame([("2025-01", "EA", "TOTAL", 100.0, None), ("2025-01", "EA", "TOTAL", 100.0, None),
                 ("2025-02", "EA", "TOTAL", 0.0, None), ("2025-03", "EA", "TOTAL", np.inf, "e"),
                 ("2025-04", "EA", "TOTAL", 101.0, "p")])
    clean, log = clean_indices(df)
    assert len(clean) == 2 and log["exact duplicates dropped"] == 1
    assert log["non-positive or non-finite removed"] == 2


def test_conflicting_duplicates_stop_the_pipeline():
    df = _frame([("2025-01", "EA", "TOTAL", 100.0, None), ("2025-01", "EA", "TOTAL", 100.5, None)])
    with pytest.raises(DataQualityError):
        clean_indices(df)


def test_negative_weights_stop_the_pipeline():
    w = pd.DataFrame({"time": ["2025"], "geo": ["EA"], "coicop": ["NRG"], "value": [-1.0]})
    with pytest.raises(DataQualityError):
        clean_weights(w, ["time", "geo", "coicop"], "item weights")


def test_series_report_counts_gaps_and_flags():
    idx = pd.DataFrame({("EA", "TOTAL"): [100.0, np.nan, 101.0, 102.0]},
                       index=pd.period_range("2025-01", periods=4, freq="M"))
    flags = _frame([("2025-04", "EA", "TOTAL", 102.0, "p")])[["time", "geo", "coicop", "flag"]]
    rep = series_report(idx, flags, None).iloc[0]
    assert rep["interior gaps"] == 1 and rep["observations"] == 3 and rep["flags"] == {"p": 1}


def test_both_partitions_reproduce_the_headline():
    checks = run_checks(make_demo_data(load_config()))
    assert checks["passed"].all()
    assert {"Main aggregates", "COICOP divisions"} <= set(checks["group"])
