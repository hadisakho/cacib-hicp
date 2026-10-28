import numpy as np
import pandas as pd
import pytest

from hicp.aggregation import mom_log, yoy
from hicp.config import load_config
from hicp.eurostat import jsonstat_to_frame
from hicp.evaluation import EQUAL_WEIGHTS, dm_test, interval_bands, selection_replay
from hicp.forecast import (ARX_BOTTOM_UP, ETS_BOTTOM_UP, ETS_HEADLINE, MEMBERS, NO_CHANGE, RECOMMENDED,
                           ForecastSetup, backtest, base_effect_decomposition, easter_regressor, final_forecast)
from hicp.passthrough import energy_passthrough, ratio_bands
from hicp.market import share_of_energy
from hicp.shocks import Shock, fuel_price_elasticity, simulate
from hicp.synthetic import make_demo_data


@pytest.fixture(scope="module")
def demo():
    return make_demo_data(load_config())


@pytest.fixture(scope="module")
def setup(demo):
    return ForecastSetup(demo.area_components(), demo.area_component_weights(), demo.headline())


def test_base_effect_identity(demo):
    """pi_T+3 - pi_T = sum of new monthly changes - sum of dropped ones (log points)."""
    I = demo.headline()
    T = pd.Period("2025-06", freq="M")
    pi, m = 100 * np.log(I / I.shift(12)), mom_log(I)
    assert pi.loc[T + 3] - pi.loc[T] == pytest.approx(sum(m.loc[T + k] - m.loc[T + k - 12] for k in (1, 2, 3)), abs=1e-10)


def test_decomposition_adds_up_to_published_rate(demo):
    I, T = demo.headline(), pd.Period("2025-03", freq="M")
    d = base_effect_decomposition(I, T, 3)
    assert d["pp"].iloc[0] + d.loc[d["kind"].isin(["new", "base"]), "pp"].sum() == pytest.approx(d["pp"].iloc[-1])
    assert d["pp"].iloc[-1] == pytest.approx(yoy(I).loc[T + 3])


@pytest.mark.parametrize("model", [ETS_HEADLINE, ETS_BOTTOM_UP, ARX_BOTTOM_UP, RECOMMENDED])
def test_no_look_ahead(setup, model):
    """Changing data after the origin must not change the forecast."""
    origin = pd.Period("2019-05", freq="M")
    base = setup.forecasters()[model](origin)
    noisy = ForecastSetup(setup.levels.copy(), setup.weights, setup.headline.copy())
    noisy.levels.loc[origin + 1:] *= 1.5
    noisy.headline.loc[origin + 1:] *= 1.5
    assert np.allclose(base, noisy.forecasters()[model](origin))


def test_recommended_is_equal_weight_average(setup):
    fcs, origin = setup.forecasters(), pd.Period("2021-02", freq="M")
    members = pd.concat([fcs[m](origin) for m in MEMBERS], axis=1)
    assert np.allclose(fcs[RECOMMENDED](origin), members.mean(axis=1))


def test_backtest_and_intervals(setup, demo):
    bt = backtest(setup.forecasters(), demo.headline(), "2023-01", end=pd.Period("2024-06", freq="M"))
    assert set(bt["h"]) == {1, 2, 3} and bt["error"].notna().all()
    bands = interval_bands(bt, RECOMMENDED)
    assert (bands["lo80"] <= bands["lo50"]).all() and (bands["hi50"] <= bands["hi80"]).all()


def test_easter_regressor_centred():
    p = pd.period_range("1950-01", "2050-12", freq="M")
    e = pd.Series(easter_regressor(p), index=p)
    assert e.groupby(e.index.month).mean().abs().max() < 1e-12
    assert e[~e.index.month.isin([3, 4])].abs().max() == 0


def test_dm_test_detects_better_model():
    rng = np.random.default_rng(3)
    stat, p = dm_test(rng.normal(0, 0.5, 200), rng.normal(0, 1.0, 200), h=3)
    assert stat < 0 and p < 0.01


def test_step_shock_has_weight_times_size_effect_for_12_months(demo):
    start = demo.last_month + 2
    res = simulate(demo, [Shock("NRG", 10, str(start))], horizon=24)
    w = demo.area_component_weights().loc[demo.last_month.year, "NRG"] / 1000
    effect = res.delta["direct"]
    # ~ weight x size for 12 months; the exact effect also moves with energy's relative price
    window = effect.loc[start:start + 11]
    assert window.between(0.85 * 10 * w, 1.05 * 10 * w).all()
    assert abs(effect.loc[start + 12]) < 0.02          # drops out (small chain-linking residual)
    assert abs(effect.loc[: start - 1]).max() < 1e-12  # nothing before the shock
    assert res.by_component.sum(axis=1).loc[start] == pytest.approx(res.delta["total"].loc[start], abs=1e-6)


def test_country_shock_scales_with_country_weight(demo):
    start = str(demo.last_month + 1)
    de = simulate(demo, [Shock("SERV", 5, start, countries=["DE"])], horizon=12)
    mt = simulate(demo, [Shock("SERV", 5, start, countries=["MT"])], horizon=12)
    assert de.delta["total"].max() > 50 * mt.delta["total"].max() > 0


def test_fuel_price_elasticity():
    # crude 80 $/bbl at 1.10 $/EUR = 0.457 EUR/l; with 0.60 excise and 0.25 margin -> 0.35
    assert fuel_price_elasticity(80, 1.10, 0.60, 0.25) == pytest.approx(0.350, abs=0.002)


def test_jsonstat_parser():
    js = {"id": ["geo", "time"], "size": [2, 2],
          "dimension": {"geo": {"category": {"index": {"EA": 0, "DE": 1}}},
                        "time": {"category": {"index": {"2025-01": 0, "2025-02": 1}}}},
          "value": {"0": 100.0, "1": 100.5, "3": 101.2}, "status": {"3": "p"}}
    df = jsonstat_to_frame(js)
    row = df[(df.geo == "DE") & (df.time == "2025-02")].iloc[0]
    assert len(df) == 3 and row.value == 101.2 and row.flag == "p"


def test_division_and_multi_component_shocks(demo):
    """Two COICOP divisions shocked in two countries: impact ~ sum of weight x size."""
    start = str(demo.last_month + 1)
    res = simulate(demo, [Shock("CP07", 10, start, countries=["FR"]), Shock("CP11", 5, start, countries=["FR", "ES"])],
                   horizon=12, codes=demo.divisions)
    y = demo.last_month.year
    cw, iw = demo.country_weights.loc[y] / 1000, demo.item_weights.loc[y] / 1000
    expected = 10 * cw["FR"] * iw[("FR", "CP07")] + 5 * (cw["FR"] * iw[("FR", "CP11")] + cw["ES"] * iw[("ES", "CP11")])
    assert res.delta["total"].loc[start] == pytest.approx(expected, rel=0.15)
    assert set(res.by_component.loc[start][lambda s: s.abs() > 1e-9].index) == {"CP07", "CP11"}
    assert res.delta["indirect"].abs().max() < 1e-12  # no indirect effects on divisions


def test_shock_on_code_outside_breakdown_is_rejected(demo):
    with pytest.raises(ValueError):
        simulate(demo, [Shock("CP07", 10, str(demo.last_month + 1))], codes=demo.components)


@pytest.mark.parametrize("partition", ["components", "divisions"])
def test_null_scenario_reproduces_the_central_forecast(setup, demo, partition):
    """Anchored baseline = recommended forecast over its horizon, for both breakdowns."""
    fc = final_forecast(setup)["yoy"]
    codes = getattr(demo, partition)
    res = simulate(demo, [], horizon=24, codes=codes, anchor=fc)
    assert np.allclose(res.yoy["baseline"].reindex(fc.index), fc, atol=1e-9)
    assert res.delta["total"].abs().max() < 1e-12 and res.forecast_end == fc.index[-1]
    # impacts barely depend on the baseline path
    shock = [Shock(codes[0], 10, str(demo.last_month + 1))]
    free = simulate(demo, shock, horizon=24, codes=codes).delta["total"]
    anchored = simulate(demo, shock, horizon=24, codes=codes, anchor=fc).delta["total"]
    assert (free - anchored).abs().max() < 0.02


def test_passthrough_sample_bounds_origins_and_outcomes(demo):
    """With sample (a, b), no observation after b is used at any horizon, even through t+h."""
    levels = demo.area_components()
    sample = ("2005-01", "2020-12")
    lp = energy_passthrough(levels, horizons=6, lags=3, sample=sample)
    noisy = levels.copy()
    rng = np.random.default_rng(0)
    after = noisy.index > pd.Period(sample[1], freq="M")
    noisy.loc[after] *= np.exp(rng.normal(0, 0.02, size=(after.sum(), noisy.shape[1])))
    lp2 = energy_passthrough(noisy, horizons=6, lags=3, sample=sample)
    for c in levels.columns:
        assert np.allclose(lp[c]["beta"], lp2[c]["beta"]) and np.allclose(lp[c]["se"], lp2[c]["se"])
    n = lp["SERV"]["n"]
    assert (n.diff().dropna() == -1).all()  # one origin fewer per horizon
    bands = ratio_bands(lp)
    assert np.allclose(bands["central"], lp["ratio"][bands["central"].columns])


def _toy_backtest(seed: int = 1) -> pd.DataFrame:
    """h=3 backtest: model A is accurate until mid-2012, model B afterwards."""
    rng = np.random.default_rng(seed)
    rows = []
    for o in pd.period_range("2010-01", "2019-12", freq="M"):
        actual, early = rng.normal(2, 1), o < pd.Period("2012-07", freq="M")
        fc = {"A": actual + rng.normal(0, 0.2 if early else 1.0), "B": actual + rng.normal(0, 1.0 if early else 0.2),
              NO_CHANGE: actual + rng.normal(0, 0.8)}
        fc[RECOMMENDED] = (fc["A"] + fc["B"]) / 2
        rows += [{"origin": o, "model": m, "h": 3, "target": o + 3, "forecast": f, "actual": actual,
                  "error": actual - f} for m, f in fc.items()]
    return pd.DataFrame(rows)


def test_selection_replay_uses_only_published_errors():
    bt = _toy_backtest()
    forecasts, summary = selection_replay(bt, 3, ["A", "B"])
    cut = pd.Period("2016-06", freq="M")
    later = bt["target"] > cut  # scramble everything published after the cut
    bt2 = bt.copy()
    bt2.loc[later, "actual"] += 5.0
    bt2["error"] = bt2["actual"] - bt2["forecast"]
    forecasts2, _ = selection_replay(bt2, 3, ["A", "B"])
    pd.testing.assert_frame_equal(forecasts.loc[:cut], forecasts2.loc[:cut])  # choices up to the cut unchanged
    assert np.allclose(forecasts[EQUAL_WEIGHTS], bt[bt.model == RECOMMENDED].set_index("origin")
                       .loc[forecasts.index, "forecast"])
    assert summary.loc["Best past RMSE", "model switches"] >= 1  # A first, B once its record is better



def test_rate_shape_compounds():
    """+3% a year on a component growing 10% a year gives 13.3%, not 13%."""
    months = pd.period_range("2026-01", "2027-12", freq="M")
    s = Shock("SERV", 3.0, "2026-01", shape="rate").path(months)
    assert s[11] == pytest.approx(0.03) and s[23] == pytest.approx(1.03 ** 2 - 1)
    assert (1.10 * (1 + s[23]) / (1 + s[11]) - 1) * 100 == pytest.approx(13.3)


def test_share_of_energy(demo):
    data = demo
    y = int(data.item_weights.index[-1])
    weights = data.item_weights.copy()
    weights[("DE", "CP0451")], weights[("DE", "CP0452")] = 0.0, 0.0
    weights.loc[y, ("DE", "CP0451")] = 0.3 * weights.loc[y, ("DE", "NRG")]
    weights.loc[y, ("DE", "CP0452")] = 0.2 * weights.loc[y, ("DE", "NRG")]
    data = type(demo)(**{**demo.__dict__, "item_weights": weights})
    share, source = share_of_energy(data, ["CP0451", "CP0452"], "DE", fallback=0.9)
    assert share == pytest.approx(0.5) and str(y) in source
    assert share_of_energy(demo, ["CP0451", "CP0452"], "DE", fallback=0.9) == (0.9, "config.yaml fallback")
