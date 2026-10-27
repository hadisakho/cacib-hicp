"""Forecast evaluation: accuracy by window, Diebold-Mariano, Hansen SPA (data snooping),
prediction intervals with volatility scaling, real-time coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .forecast import NO_CHANGE


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
    """Honest interval check: at each origin, build the interval from errors already
    observed (targets <= origin) and record whether the outcome falls inside."""
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
