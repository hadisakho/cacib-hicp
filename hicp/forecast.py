"""Three-month-ahead forecast of euro area headline HICP inflation.

Key identity (pi = 12-month log change, m = month-on-month log change):

    pi_T+h = pi_T + sum_{k=1..h} (m_T+k - m_T+k-12)

The m_T+k-12 terms are already published (base effects), so forecasting the annual rate
h months ahead = forecasting h non-seasonally-adjusted monthly changes.

Recommended model (chosen on robustness across 2013-2026, see README): the equal-weight
average of four models from two families -
    ETS on the headline index            ETS on each main component, re-aggregated
    seasonal ARX on each component       seasonal median on each component
Component paths are re-aggregated with the chain-linking engine and anchored to the
published headline index.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd
from dateutil.easter import easter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .aggregation import chain_link, dec, lag, mom_log, price_update, yoy

Forecaster = Callable[[pd.Period], pd.Series]  # origin -> YoY forecasts (%) for origin+1..origin+h

NO_CHANGE = "No change (random walk)"
SEASONAL_NAIVE = "Seasonal naive"
ETS_HEADLINE = "ETS headline"
ETS_BOTTOM_UP = "ETS bottom-up"
ARX_BOTTOM_UP = "Seasonal ARX bottom-up"
MEDIAN_BOTTOM_UP = "Seasonal median bottom-up"
RECOMMENDED = "4-model average (recommended)"
MEMBERS = [ETS_HEADLINE, ETS_BOTTOM_UP, ARX_BOTTOM_UP, MEDIAN_BOTTOM_UP]
BENCHMARKS = [NO_CHANGE, SEASONAL_NAIVE]


# ---- calendar regressor -----------------------------------------------------------

@lru_cache(maxsize=None)
def _easter_share(year: int, month: int, window: int = 8) -> float:
    """Share of the `window` days before Easter Sunday that fall in (year, month)."""
    e = easter(year)
    return sum((e - timedelta(days=k)).month == month for k in range(1, window + 1)) / window


@lru_cache(maxsize=None)
def _easter_mean(month: int) -> float:
    return float(np.mean([_easter_share(y, month) for y in range(1950, 2051)]))


def easter_regressor(periods: pd.PeriodIndex) -> np.ndarray:
    """Easter effect (package holidays, accommodation), centred per calendar month."""
    return np.array([_easter_share(p.year, p.month) - _easter_mean(p.month) for p in periods])


# ---- component models: monthly log changes (x100) in, h forecasts out ----------------

class SeasonalMedian:
    """Median change of the same calendar month over the last `years` years. With
    `trend_months`, keep the seasonal shape but set its level to the mean change of the
    last `trend_months` months (recent momentum held) - used for shock baselines."""

    def __init__(self, years: int = 6, trend_months: int | None = None):
        self.years, self.trend_months = years, trend_months

    def __call__(self, m: pd.Series, h: int) -> np.ndarray:
        m = m.dropna()
        T = m.index[-1]
        hist = m[m.index > T - 12 * self.years]
        med = hist.groupby(hist.index.month).median()
        if self.trend_months:
            med = med - med.mean() + m.iloc[-self.trend_months:].mean()
        return med.reindex(pd.period_range(T + 1, periods=h, freq="M").month).to_numpy()


class SeasonalARX:
    """m_t = sum_s d_s D_s,t + g Easter_t + phi m_t-1 + e_t, OLS on a rolling window,
    iterated forecasts."""

    def __init__(self, window: int = 120):
        self.window = window

    @staticmethod
    def _design(periods: pd.PeriodIndex, ylag: np.ndarray) -> np.ndarray:
        dummies = [(periods.month == s).astype(float) for s in range(1, 13)]
        return np.column_stack([*dummies, easter_regressor(periods), ylag])

    def __call__(self, m: pd.Series, h: int) -> np.ndarray:
        m = m.dropna()
        periods = m.index[-self.window:]
        X = self._design(periods, lag(m, 1).reindex(periods).to_numpy())
        y = m.reindex(periods).to_numpy()
        ok = np.isfinite(X).all(axis=1)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        last, out = m.iloc[-1], []
        for k in range(1, h + 1):
            last = (self._design(pd.PeriodIndex([m.index[-1] + k], freq="M"), np.array([last])) @ beta).item()
            out.append(last)
        return np.array(out)


def ets_levels(levels: np.ndarray, h: int) -> np.ndarray:
    """Holt-Winters ETS: additive damped trend + additive monthly seasonality."""
    model = ExponentialSmoothing(levels, trend="add", damped_trend=True, seasonal="add",
                                 seasonal_periods=12, initialization_method="estimated")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(optimized=True, use_brute=False)
        if not fit.mle_retvals.get("success", False):
            fit = model.fit(optimized=True, use_brute=True)
    return np.asarray(fit.forecast(h))


class ETS:
    """ETS in the component interface: levels rebuilt from the monthly changes (the model
    is scale-equivariant), last `window` months; seasonal median if the fit fails."""

    def __init__(self, window: int = 120):
        self.window = window

    def __call__(self, m: pd.Series, h: int) -> np.ndarray:
        m = m.dropna()
        lv = 100 * np.exp(np.concatenate([[0.0], np.cumsum(m.to_numpy()) / 100]))[-self.window:]
        try:
            f = ets_levels(lv, h)
        except Exception:
            return SeasonalMedian()(m, h)
        return 100 * np.diff(np.log(np.concatenate([[lv[-1]], f])))


# ---- bottom-up machinery -------------------------------------------------------------

def forecast_levels(levels: pd.DataFrame, model: Callable, h: int) -> pd.DataFrame:
    """Index levels of each component for the h months after the last row."""
    T = levels.index[-1]
    out = {}
    for comp in levels.columns:
        f = model(mom_log(levels[comp]).dropna(), h)
        out[comp] = levels[comp].iloc[-1] * np.exp(np.cumsum(f) / 100.0)
    return pd.DataFrame(out, index=pd.period_range(T + 1, periods=h, freq="M"))


def weights_known_at(weights: pd.DataFrame, origin: pd.Period, years: range, levels: pd.DataFrame) -> pd.DataFrame:
    """Official weights up to the origin's year (published with January data), then
    price-updated proxies for later years."""
    w = weights[weights.index <= origin.year].copy()
    for y in years:
        if y not in w.index:
            prev = w.loc[y - 1, levels.columns]
            w.loc[y] = 1000 * price_update(prev / prev.sum(), levels.loc[dec(y - 1)], levels.loc[dec(y - 2)])
    return w.sort_index()


def bottom_up_yoy(levels: pd.DataFrame, weights: pd.DataFrame, headline: pd.Series, model: Callable,
                  origin: pd.Period, h: int) -> tuple[pd.Series, pd.DataFrame]:
    """Bottom-up headline YoY for origin+1..origin+h from data <= origin. The re-aggregated
    path is applied as growth to the *published* headline, so base months are official."""
    hist = levels.loc[:origin]
    fc = forecast_levels(hist, model, h)
    full = pd.concat([hist, fc])
    w = weights_known_at(weights, origin, range(origin.year, (origin + h).year + 1), full)
    agg = chain_link(full.loc[dec(origin.year - 3):], w, rebase_year=None)
    pub = headline.loc[:origin]
    path = pub.loc[origin] * agg.loc[fc.index] / agg.loc[origin]
    return 100 * (path / pub.reindex(fc.index - 12).to_numpy() - 1), fc


@dataclass
class ForecastSetup:
    levels: pd.DataFrame   # euro area component indices
    weights: pd.DataFrame  # euro area component weights (per mille)
    headline: pd.Series    # published euro area headline index
    h: int = 3
    window: int = 120
    seasonal_years: int = 6

    def _targets(self, origin: pd.Period) -> pd.PeriodIndex:
        return pd.period_range(origin + 1, periods=self.h, freq="M")

    def _from_index(self, origin: pd.Period, path: np.ndarray) -> pd.Series:
        idx = self._targets(origin)
        return pd.Series(100 * (path / self.headline.reindex(idx - 12).to_numpy() - 1), index=idx)

    def forecasters(self) -> dict[str, Forecaster]:
        H = self.headline

        def no_change(origin):
            return pd.Series(yoy(H.loc[:origin]).loc[origin], index=self._targets(origin))

        def seasonal_naive(origin):  # average path of the same months over the past 5 years
            I = H.loc[:origin]
            return self._from_index(origin, np.array([
                I.iloc[-1] * np.mean([I.loc[t - 12 * y] / I.loc[origin - 12 * y] for y in range(1, 6)])
                for t in self._targets(origin)]))

        def ets_headline(origin):
            return self._from_index(origin, ets_levels(H.loc[:origin].iloc[-self.window:].to_numpy(), self.h))

        def bottom_up(model):
            return lambda origin: bottom_up_yoy(self.levels, self.weights, H, model, origin, self.h)[0]

        out = {NO_CHANGE: no_change, SEASONAL_NAIVE: seasonal_naive, ETS_HEADLINE: ets_headline,
               ETS_BOTTOM_UP: bottom_up(ETS(self.window)),
               ARX_BOTTOM_UP: bottom_up(SeasonalARX(self.window)),
               MEDIAN_BOTTOM_UP: bottom_up(SeasonalMedian(self.seasonal_years))}
        out = {k: lru_cache(maxsize=None)(f) for k, f in out.items()}  # the average reuses them
        out[RECOMMENDED] = lambda origin: pd.concat([out[m](origin) for m in MEMBERS], axis=1).mean(axis=1)
        return out


def backtest(forecasters: dict[str, Forecaster], headline: pd.Series, start: str,
             end: pd.Period | None = None) -> pd.DataFrame:
    """Pseudo-out-of-sample evaluation: every forecast uses data up to its origin only.
    One row per (origin, model, horizon)."""
    actual = yoy(headline)
    origins = pd.period_range(pd.Period(start, freq="M"), end or headline.dropna().index[-1] - 1, freq="M")
    rows = []
    for origin in origins:
        for name, f in forecasters.items():
            for k, (target, value) in enumerate(f(origin).items(), start=1):
                rows.append({"origin": origin, "model": name, "h": k, "target": target,
                             "forecast": value, "actual": actual.get(target, np.nan)})
    bt = pd.DataFrame(rows)
    bt["error"] = bt["actual"] - bt["forecast"]
    return bt


def base_effect_decomposition(index: pd.Series, origin: pd.Period, h: int) -> pd.DataFrame:
    """Change in the annual rate from origin to origin+h, split into new monthly changes
    (m_T+k) and base effects (-m_T+k-12). Exact in log points, rescaled proportionally so
    the terms add up to the change in the published-style rate (pp)."""
    m, rate = mom_log(index), yoy(index)
    terms = []
    for k in range(1, h + 1):
        t = origin + k
        terms.append({"term": f"{t.strftime('%b %Y')}<br>new", "kind": "new", "log": m.loc[t]})
        terms.append({"term": f"{(t - 12).strftime('%b %Y')}<br>drops out", "kind": "base", "log": -m.loc[t - 12]})
    df = pd.DataFrame(terms)
    change = rate.loc[origin + h] - rate.loc[origin]
    df["pp"] = df["log"] * (change / df["log"].sum()) if df["log"].sum() != 0 else 0.0
    start = {"term": f"{origin.strftime('%b %Y')}<br>YoY", "kind": "start", "log": np.nan, "pp": rate.loc[origin]}
    end = {"term": f"{(origin + h).strftime('%b %Y')}<br>YoY", "kind": "end", "log": np.nan, "pp": rate.loc[origin + h]}
    return pd.DataFrame([start, *df.to_dict("records"), end])


def final_forecast(setup: ForecastSetup, origin: pd.Period | None = None) -> dict:
    """Forecasts of every model from the latest data, plus the headline index implied by
    the recommended model (for the base-effect decomposition)."""
    origin = origin or setup.headline.dropna().index[-1]
    fcs = setup.forecasters()
    models = pd.DataFrame({k: f(origin) for k, f in fcs.items()})
    path = models[RECOMMENDED]
    pub = setup.headline.loc[:origin]
    implied = pub.reindex(path.index - 12).to_numpy() * (1 + path.to_numpy() / 100)
    return {"origin": origin, "models": models, "yoy": path,
            "index": pd.concat([pub, pd.Series(implied, index=path.index)])}
