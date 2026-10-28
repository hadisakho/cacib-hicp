"""Indirect pass-through of energy prices to other HICP components (local projections).

For component j and horizon h = 0..H (Jorda, 2005):

    100*[ln I_j,t+h - ln I_j,t-1] = a_h + b_h x_t + sum_l c_l x_t-l + sum_l d_l dy_j,t-l
                                    + month dummies + u_t+h

with x_t the monthly log change of the energy HICP. Standard errors are HAC (Newey-West,
h+1 lags) because the overlapping left-hand side makes u_t+h autocorrelated.

The regressor is the observed energy-price change. The estimates describe historical associations
rather than identified causal effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .aggregation import lag


def local_projection(y_level: pd.Series, x: pd.Series, horizons: int = 24, lags: int = 12,
                     own_lags: bool = True, sample: tuple[str, str] | None = None) -> pd.DataFrame:
    """Cumulative response (%) of `y_level` to a 1-log-point move in `x` at t.

    own_lags : include lags of the component's own monthly change (switch off when y is
               the source itself, whose lags are already the x lags)
    sample   : (first, last) month of the data used. Both origins t and outcomes t+h stay
               inside it: at horizon h the last origin is `last - h`, so e.g. ("1997-01",
               "2020-12") really excludes 2021 onwards at every horizon.
    """
    ly = 100 * np.log(y_level.dropna())
    dy = (ly - lag(ly, 1))
    idx = ly.index
    x = x.reindex(idx)
    controls = {f"x_l{j}": lag(x, j) for j in range(1, lags + 1)}
    if own_lags:
        controls |= {f"dy_l{j}": lag(dy, j) for j in range(1, lags + 1)}
    months = pd.get_dummies(pd.Series(idx.month, index=idx), prefix="m", drop_first=True, dtype=float)
    base = pd.concat([x.rename("x"), pd.DataFrame(controls), months], axis=1)

    rows = []
    for h in range(horizons + 1):
        lead = ly.reindex(idx + h)
        lead.index = idx
        lhs = (lead - lag(ly, 1)).rename("lhs")
        df = pd.concat([lhs, base], axis=1).dropna()
        if sample is not None:
            df = df.loc[pd.Period(sample[0], freq="M"):pd.Period(sample[1], freq="M") - h]
        res = sm.OLS(df["lhs"], sm.add_constant(df.drop(columns="lhs"))) \
            .fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        b, se = res.params["x"], res.bse["x"]
        rows.append({"h": h, "beta": b, "se": se, "lo": b - 1.645 * se, "hi": b + 1.645 * se, "n": int(res.nobs)})
    return pd.DataFrame(rows).set_index("h")


def energy_passthrough(levels: pd.DataFrame, energy: str = "NRG", horizons: int = 24,
                       lags: int = 12, sample: tuple[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Local projections of every component on energy price changes.

    Also returns 'ratio': response of j divided by energy's own cumulative response -
    the pass-through to j per 1% *sustained* rise in energy prices, which is what the
    shock engine needs.
    """
    x = 100 * np.log(levels[energy] / lag(levels[energy], 1))
    out = {c: local_projection(levels[c], x, horizons, lags, own_lags=(c != energy), sample=sample)
           for c in levels.columns}
    own = out[energy]["beta"]
    out["ratio"] = pd.DataFrame({c: out[c]["beta"] / own for c in levels.columns if c != energy})
    return out


def ratio_bands(lp: dict[str, pd.DataFrame], energy: str = "NRG") -> dict[str, pd.DataFrame]:
    """Pass-through ratios (central, low, high) for the shock engine: each component's
    response and its 90% band divided by energy's own cumulative response. A sensitivity
    range on the component coefficients, not a full confidence interval."""
    own = lp[energy]["beta"]
    comps = [c for c in lp if c not in (energy, "ratio")]
    return {name: pd.DataFrame({c: lp[c][col] / own for c in comps})
            for name, col in (("central", "beta"), ("low", "lo"), ("high", "hi"))}
