"""Shock simulator: pass-through of component / country price shocks to euro area
headline inflation.

A shock is a deviation of a price level from its baseline path,
    I_shock(c,i,t) = I_base(c,i,t) * (1 + s(c,i,t)),
applied to country x component series and re-aggregated *exactly* with the euro area
chain-linking (effective weights omega_c,i = country weight x item weight). Two layers:

  direct    the shocked components themselves (pure accounting, no estimation);
  indirect  energy shocks spilling into food, goods and services, using pass-through
            ratios estimated by local projections (`passthrough.energy_passthrough`).

First-order intuition: a permanent step of x% on a basket share omega lifts headline
YoY by ~omega*x for 12 months, then drops out (base effect).

Assumptions: weights after the last published year are fixed; indirect
pass-through is the same in every country; no second-round (wage) effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aggregation import chain_link, contributions, dec, effective_weights, mom_log, yoy
from .data import HICPData
from .forecast import SeasonalMedian

SHAPES = ("step", "ramp", "temporary", "rate")


@dataclass
class Shock:
    """component : COICOP code of one of the configured components (e.g. "NRG")
    size       : % change in the price level (for shape="rate": extra annual growth of
                 the price level, %, compounded)
    start      : first affected month, e.g. "2026-10"
    countries  : list of geo codes; None = every euro area member
    shape      : step | ramp (reaches `size` after `ramp_months`) |
                 temporary (lasts `duration` months) | rate (the level grows `size`% a
                 year faster, compounded: about pi + size at low inflation, but 13.3%
                 rather than 13% for pi = 10% and size = 3)
    """
    component: str
    size: float
    start: str
    countries: list[str] | None = None
    shape: str = "step"
    ramp_months: int = 6
    duration: int = 6

    def path(self, periods: pd.PeriodIndex) -> np.ndarray:
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        start = pd.Period(self.start, freq="M")
        k = np.array([(p - start).n for p in periods])  # months since start
        on = k >= 0
        x = self.size / 100
        if self.shape == "step":
            s = np.where(on, x, 0.0)
        elif self.shape == "ramp":
            s = np.where(on, x * np.minimum(1.0, (k + 1) / self.ramp_months), 0.0)
        elif self.shape == "temporary":
            s = np.where(on & (k < self.duration), x, 0.0)
        else:
            s = np.where(on, (1 + x) ** ((k + 1) / 12) - 1, 0.0)
        return s


@dataclass
class ShockResult:
    yoy: pd.DataFrame           # baseline / shocked / direct-only headline YoY (%)
    delta: pd.DataFrame         # total, direct, indirect impact (pp)
    by_component: pd.DataFrame  # impact by component (pp), direct + indirect
    by_country: pd.DataFrame    # impact by country (pp)
    paths: pd.DataFrame         # price-level deviations applied, (geo, coicop) columns
    forecast_end: pd.Period | None = None  # last month the baseline equals the central forecast


LITRES_PER_BARREL = 158.987


def fuel_price_elasticity(brent_usd: float, usd_per_eur: float, excise_eur: float, margin_eur: float) -> float:
    """Elasticity of the pump price to the crude price.

    P = (c + excise + margin) * (1 + VAT)  =>  dlnP / dln c = c / (c + excise + margin),
    c = crude cost per litre in EUR. VAT is proportional and cancels; excise is a fixed
    amount per litre, which is why only about a third to a half of a crude move reaches the pump.
    """
    crude = brent_usd / usd_per_eur / LITRES_PER_BARREL
    return crude / (crude + excise_eur + margin_eur)


def baseline(data: HICPData, horizon: int, codes: list[str] | None = None, years: int = 6,
             anchor: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country x code index levels extended `horizon` months, and euro area effective
    weights extended by holding the last published year fixed. `codes` = a partition of
    the headline (main aggregates by default, or COICOP divisions).

    Path: usual seasonal pattern with each series' last-12-month momentum. With `anchor`
    (the central forecast of headline YoY, %, for the first months), every series is
    rescaled proportionally month by month so the aggregate reproduces it exactly: a
    scenario without shocks then equals the forecast. Months after the anchor are a
    conditional assumption, not a forecast."""
    codes = codes or data.components
    T = data.last_month
    end = T + horizon
    hist = data.country_breakdown(codes).loc[dec(T.year - 3):T].dropna(axis=1, how="all")
    model = SeasonalMedian(years, trend_months=12)
    levels = hist.reindex(pd.period_range(hist.index[0], end, freq="M"))
    for col in hist.columns:  # each series continues from its own last observation
        s = data.indices[col].loc[:T].dropna()
        n = (end - s.index[-1]).n
        f = s.iloc[-1] * np.exp(np.cumsum(model(mom_log(s).dropna(), n)) / 100)
        levels.loc[s.index[-1] + 1:, col] = f
    omega = effective_weights(data.item_weights, data.country_weights, data.countries, codes)
    omega = omega.reindex(columns=levels.columns).fillna(0.0)
    for y in range(int(omega.index.max()) + 1, end.year + 1):
        omega.loc[y] = omega.loc[y - 1]
    omega = omega.sort_index()
    if anchor is not None:
        for t, rate in anchor.items():  # in order: a common factor from t onwards keeps relative prices
            t = pd.Period(t, freq="M")
            agg = chain_link(levels, omega, rebase_year=None)
            levels.loc[t:] *= agg.loc[t - 12] * (1 + rate / 100) / agg.loc[t]
    return levels, omega


def indirect_path(source: np.ndarray, ratio: pd.Series) -> np.ndarray:
    """Convolve the change in the source shock with the cumulative pass-through ratio:
    s_j,t = sum_tau PT_j(t - tau) * (s_src,tau - s_src,tau-1); PT held at its last
    horizon beyond the estimated range."""
    pt = ratio.to_numpy()
    d = np.diff(np.concatenate([[0.0], source]))
    out = np.zeros_like(source)
    for tau in np.flatnonzero(d):
        horizon = np.arange(len(source) - tau)
        out[tau:] += d[tau] * pt[np.minimum(horizon, len(pt) - 1)]
    return out


def simulate(data: HICPData, shocks: list[Shock], horizon: int = 24,
             passthrough_ratio: pd.DataFrame | None = None, energy: str = "NRG",
             codes: list[str] | None = None, anchor: pd.Series | None = None) -> ShockResult:
    """Apply `shocks` (one or more components and/or countries) and re-aggregate exactly.
    `codes` chooses the partition (default: main aggregates). Indirect effects need the
    energy aggregate in the partition and a pass-through table. `anchor`: central forecast
    (headline YoY by month) that the baseline reproduces over the forecast horizon."""
    codes = codes or data.components
    if unknown := {sh.component for sh in shocks} - set(codes):
        raise ValueError(f"Shocked codes {sorted(unknown)} are not in the chosen breakdown")
    levels, omega = baseline(data, horizon, codes, anchor=anchor)
    periods = levels.index
    T = data.last_month
    members = list(dict.fromkeys(levels.columns.get_level_values("geo")))

    direct = pd.DataFrame(0.0, index=periods, columns=levels.columns)
    for sh in shocks:
        path = sh.path(periods)
        for geo in (sh.countries or members):
            if (geo, sh.component) in direct.columns:
                direct[(geo, sh.component)] = (1 + direct[(geo, sh.component)]) * (1 + path) - 1

    indirect = pd.DataFrame(0.0, index=periods, columns=levels.columns)
    if (passthrough_ratio is not None and energy in codes
            and (direct.xs(energy, axis=1, level="coicop") != 0).any().any()):
        w_nrg = omega.iloc[-1].xs(energy, level="coicop")
        src = (direct.xs(energy, axis=1, level="coicop") * w_nrg).sum(axis=1) / w_nrg.sum()
        for comp in passthrough_ratio.columns:
            pth = indirect_path(src.to_numpy(), passthrough_ratio[comp].dropna())
            for geo in members:
                if (geo, comp) in indirect.columns:
                    indirect[(geo, comp)] = pth

    total = (1 + direct) * (1 + indirect) - 1
    base_agg = chain_link(levels, omega, rebase_year=None)
    shock_agg = chain_link(levels * (1 + total), omega, rebase_year=None)
    direct_agg = chain_link(levels * (1 + direct), omega, rebase_year=None)

    window = pd.period_range(T - 12, periods[-1], freq="M")
    yoys = pd.DataFrame({"baseline": yoy(base_agg), "shocked": yoy(shock_agg),
                         "direct only": yoy(direct_agg)}).reindex(window)
    delta = pd.DataFrame({"total": yoys["shocked"] - yoys["baseline"],
                          "direct": yoys["direct only"] - yoys["baseline"]})
    delta["indirect"] = delta["total"] - delta["direct"]

    c_base = contributions(levels, omega, base_agg, periods=window)
    c_shock = contributions(levels * (1 + total), omega, shock_agg, periods=window)
    diff = c_shock - c_base
    by_comp = diff.T.groupby(level="coicop").sum().T.reindex(columns=codes)
    by_country = diff.T.groupby(level="geo").sum().T
    by_country = by_country.loc[:, by_country.abs().max() > 1e-6]
    end = pd.Period(anchor.index[-1], freq="M") if anchor is not None else None
    return ShockResult(yoy=yoys, delta=delta, by_component=by_comp, by_country=by_country,
                       paths=total.loc[:, (total != 0).any()], forecast_end=end)
