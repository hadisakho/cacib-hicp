import numpy as np
import pandas as pd
import pytest

from hicp.aggregation import chain_link, contributions, effective_weights, yoy


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    periods = pd.period_range("2015-12", "2020-12", freq="M")
    comps = list("ABCDE")
    idx = pd.DataFrame(np.cumprod(1 + rng.normal(0.002, 0.01, (len(periods), len(comps))), axis=0),
                       index=periods, columns=comps)
    w = pd.DataFrame(rng.dirichlet(np.ones(len(comps)), size=5) * 1000, index=range(2016, 2021), columns=comps)
    return idx, w


def test_chain_link_matches_definition(toy):
    idx, w = toy
    agg = chain_link(idx, w, rebase_year=None)
    t, D = pd.Period("2018-05", freq="M"), pd.Period("2017-12", freq="M")
    wn = w.loc[2018] / w.loc[2018].sum()
    expected = agg[D] * float((idx.loc[t] / idx.loc[D]) @ wn)
    assert agg[t] == pytest.approx(expected, rel=1e-12)


def test_contributions_sum_exactly_to_aggregate_rate(toy):
    idx, w = toy
    agg = chain_link(idx, w, rebase_year=None)
    c = contributions(idx, w, agg)
    diff = (c.sum(axis=1) - yoy(agg)).dropna()
    assert len(diff) > 30
    assert diff.abs().max() < 1e-10


def test_yoy_is_not_weighted_mean_of_component_rates(toy):
    """Why the exact decomposition is needed: the naive formula is off."""
    idx, w = toy
    agg = chain_link(idx, w, rebase_year=None)
    t = pd.Period("2019-06", freq="M")
    naive = float(yoy(idx).loc[t] @ (w.loc[2019] / w.loc[2019].sum()))
    assert abs(naive - yoy(agg).loc[t]) > 1e-6


def test_rebasing_does_not_change_rates(toy):
    idx, w = toy
    a = yoy(chain_link(idx, w, rebase_year=None)).dropna()
    b = yoy(chain_link(idx * 3.7, w, rebase_year=2019)).dropna()
    assert np.allclose(a, b)


def test_two_level_aggregation_equals_flat():
    """Country x component aggregation with effective weights = aggregating each country
    first, then countries with country weights (December-linked)."""
    rng = np.random.default_rng(1)
    periods = pd.period_range("2017-12", "2020-12", freq="M")
    countries, comps = ["X", "Y", "Z"], ["A", "B"]
    cols = pd.MultiIndex.from_product([countries, comps], names=["geo", "coicop"])
    idx = pd.DataFrame(np.cumprod(1 + rng.normal(0.002, 0.01, (len(periods), len(cols))), axis=0),
                       index=periods, columns=cols)
    years = range(2018, 2021)
    iw = pd.DataFrame(rng.uniform(100, 900, (3, len(cols))), index=years, columns=cols)
    cw = pd.DataFrame(rng.uniform(100, 900, (3, 3)), index=years, columns=countries)
    omega = effective_weights(iw, cw, countries, comps)
    flat = chain_link(idx, omega, rebase_year=None)
    by_country = pd.DataFrame({c: chain_link(idx[c], iw[c], rebase_year=None) for c in countries})
    two_level = chain_link(by_country, cw, rebase_year=None)
    assert np.allclose(flat, two_level, rtol=1e-12)


def test_country_outside_area_is_ignored(toy):
    idx, w = toy
    w2 = w.copy()
    w2.loc[2016:2017, "E"] = 0.0          # "E" joins in 2018
    idx2 = idx.copy()
    idx2.loc[:"2016-06", "E"] = np.nan    # and has no data before mid-2016
    agg = chain_link(idx2, w2, rebase_year=None)
    assert agg.notna().all() and agg.index[-1] == idx.index[-1]
