"""Forecast evaluation: accuracy by window, Diebold-Mariano, Hansen SPA (data snooping),
prediction intervals with volatility scaling, real-time coverage, month-by-month replay of
model-selection rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .forecast import NO_CHANGE, RECOMMENDED


def dm_test(e_model, e_bench, h: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal squared-error loss with the Harvey-Leybourne-Newbold
    correction; h-step errors overlap, so the long-run variance uses h-1 autocovariances.
    Negative statistic = model more accurate. Returns (statistic, two-sided p-value)."""
    d = np.asarray(e_model, float) ** 2 - np.asarray(e_bench, float) ** 2
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[: T - k] - dbar)) / T for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / T
    if var <= 0:  # Bartlett weights, always positive
        var = (gam[0] + 2 * sum((1 - k / h) * g for k, g in enumerate(gam[1:], start=1))) / T
    hln = dbar / np.sqrt(var) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(hln), float(2 * stats.t.cdf(-abs(hln), df=T - 1))


def _window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["target"] >= pd.Period(start, freq="M")]
    if end:
        df = df[df["target"] <= pd.Period(end, freq="M")]
    return df


def accuracy_table(bt: pd.DataFrame, h: int, benchmark: str = NO_CHANGE,
                   start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """RMSE, MAE, bias and DM test vs the benchmark at horizon h over a target window."""
    df = _window(bt[bt["h"] == h].dropna(subset=["actual"]), start, end)
    wide = df.pivot(index="target", columns="model", values="error")
    bench = wide[benchmark]
    rows = []
    for model in wide.columns:
        e = wide[model]
        stat, p = (np.nan, np.nan) if model == benchmark else dm_test(e, bench, h)
        rmse = np.sqrt(np.mean(e ** 2))
        rows.append({"model": model, "n": int(e.notna().sum()), "RMSE": rmse, "MAE": e.abs().mean(),
                     "bias (fc - actual)": -e.mean(), "rel. RMSE": rmse / np.sqrt(np.mean(bench ** 2)), "DM p-value": p})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def accuracy_by_window(bt: pd.DataFrame, h: int, windows: dict[str, tuple]) -> pd.DataFrame:
    """Long table: one accuracy_table per evaluation window."""
    return pd.concat([accuracy_table(bt, h, start=a, end=b).assign(window=name)
                      for name, (a, b) in windows.items()], ignore_index=True)


def spa_test(bt: pd.DataFrame, h: int, candidates: list[str], benchmark: str = NO_CHANGE,
             start: str | None = None, end: str | None = None, seed: int = 7) -> float:
    """Hansen (2005) Superior Predictive Ability test. H0: no candidate beats the
    benchmark. Accounts for having searched over all candidates (data snooping).
    Returns the consistent p-value."""
    from arch.bootstrap import SPA
    df = _window(bt[bt["h"] == h].dropna(subset=["actual"]), start, end)
    losses = (df.pivot(index="target", columns="model", values="error")[[benchmark, *candidates]] ** 2).dropna()
    spa = SPA(losses[benchmark], losses[candidates], reps=5000, block_size=6, bootstrap="stationary", seed=seed)
    spa.compute()
    return float(spa.pvalues["consistent"])


def _scale(past: pd.Series) -> float:
    """Widen intervals when the last 12 errors are more volatile than all past errors
    (never narrow them)."""
    return max(1.0, np.sqrt(np.mean(past.iloc[-12:] ** 2)) / np.sqrt(np.mean(past ** 2)))


def interval_bands(bt: pd.DataFrame, model: str, levels=(0.5, 0.8, 0.9), scale: bool = True) -> pd.DataFrame:
    """Central intervals around the point forecast from empirical quantiles of past
    out-of-sample errors, by horizon (rows) - lo50/hi50/lo80/... columns, in pp."""
    df = bt[bt["model"] == model].dropna(subset=["error"]).sort_values("target")
    out = {}
    for h, g in df.groupby("h"):
        k = _scale(g["error"]) if scale else 1.0
        out[h] = {f"{side}{int(100 * lvl)}": k * g["error"].quantile(q)
                  for lvl in levels for side, q in (("lo", (1 - lvl) / 2), ("hi", 1 - (1 - lvl) / 2))}
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("h")


def realtime_coverage(bt: pd.DataFrame, model: str, h: int, level: float = 0.8, scale: bool = True,
                      min_obs: int = 36) -> dict:
    """Evaluate interval coverage using errors available at each forecast origin."""
    df = bt[(bt["model"] == model) & (bt["h"] == h)].sort_values("origin")
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    hits = []
    for _, row in df.dropna(subset=["actual"]).iterrows():
        past = df[df["target"] <= row["origin"]].sort_values("target")["error"].dropna()
        if len(past) < min_obs:
            continue
        k = _scale(past) if scale else 1.0
        hits.append(k * past.quantile(lo_q) <= row["error"] <= k * past.quantile(hi_q))
    return {"model": model, "h": h, "level": level, "scaled": scale,
            "coverage": float(np.mean(hits)) if hits else np.nan, "n": len(hits)}


# ---- model selection, replayed in real time ------------------------------------------

EQUAL_WEIGHTS = "Equal weights, fixed (rule used)"


def selection_replay(bt: pd.DataFrame, h: int, candidates: list[str], benchmark: str = NO_CHANGE,
                     combination: str = RECOMMENDED, windows: dict[str, tuple] | None = None,
                     min_errors: int = 24, recent: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay written-down selection rules month by month. At origin t a rule only sees the
    h-step errors whose target is already published (target <= t), so the choice of model is
    itself out-of-sample, not only the forecasts. Rules start once `min_errors` errors are known.

    Rules: equal weights on `candidates` (the combination, nothing estimated); inverse-MSE
    weights; the candidate with the lowest past RMSE (all past errors, or the last `recent`);
    the lowest past RMSE among every model in the backtest (benchmarks and combination too).
    Returns (forecasts by rule, indexed by origin; summary: RMSE relative to the benchmark
    by target window, DM p-value vs the benchmark, number of model switches)."""
    df = bt[bt["h"] == h].dropna(subset=["actual"])
    fc = df.pivot(index="origin", columns="model", values="forecast")
    err = df.pivot(index="origin", columns="model", values="error")
    actual = df.groupby("origin")["actual"].first()
    everyone = list(err.columns)
    pick_rules = {"Best past RMSE": (candidates, None),
                  f"Best past RMSE, last {recent} months": (candidates, recent),
                  "Best past RMSE, any model": (everyone, None)}
    out, picks = {}, {name: {} for name in pick_rules}
    for t in fc.index:
        known = err.loc[: t - h]  # errors of forecasts whose target <= t
        if len(known) < min_errors:
            continue
        mse = (known[candidates] ** 2).mean()
        row = {EQUAL_WEIGHTS: fc.loc[t, combination],
               "Inverse-MSE weights": (fc.loc[t, candidates] / mse).sum() / (1 / mse).sum()}
        for name, (models, window) in pick_rules.items():
            past = known[models] if window is None else known[models].iloc[-window:]
            best = (past ** 2).mean().idxmin()
            row[name], picks[name][t] = fc.loc[t, best], best
        out[t] = row
    forecasts = pd.DataFrame.from_dict(out, orient="index").rename_axis("origin")
    errors = (-forecasts).add(actual.reindex(forecasts.index), axis=0)  # actual - forecast, as in the backtest
    bench = err[benchmark].reindex(forecasts.index)
    target = forecasts.index + h
    rows = []
    for rule in forecasts.columns:
        row = {"rule": rule}
        for name, (a, b) in (windows or {"full": (None, None)}).items():
            m = np.ones(len(target), bool)
            if a:
                m &= target >= pd.Period(a, freq="M")
            if b:
                m &= target <= pd.Period(b, freq="M")
            row[name] = np.sqrt(np.mean(errors.loc[m, rule] ** 2) / np.mean(bench[m] ** 2))
        row["n"] = len(target)
        row["DM p-value"] = dm_test(errors[rule], bench, h)[1]
        chosen = pd.Series(picks.get(rule, {}), dtype=object)
        row["model switches"] = int((chosen != chosen.shift()).sum() - 1) if len(chosen) else 0
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("rule")
    summary.attrs["first target"] = str(target[0])
    return forecasts, summary

