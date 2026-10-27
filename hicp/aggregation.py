"""Chain-linked HICP aggregation and exact contributions to annual rates.

The HICP is an annually chain-linked Laspeyres-type index with December of the previous
year as link month. For month t of year y, with D = Dec(y-1):

    I_t / I_D = sum_i w_i^y * I_i,t / I_i,D            (w^y normalised to sum to 1)

So the annual rate of an aggregate is NOT the weighted mean of the components' annual
rates: the 12-month window straddles two weight sets. `contributions` implements the
exact decomposition (Ribe) whose terms add up to the aggregate's annual rate:

    C_i,t = (I_D / I_t-12) * w_i^y * (I_i,t / I_i,D - 1)
          + (I_D' / I_t-12) * w_i^(y-1) * (I_i,D - I_i,t-12) / I_i,D'      D' = Dec(y-2)

The euro area is aggregated the same way with effective weights
omega_c,i = country weight_c * item weight_c,i (`effective_weights`).

Conventions: indices are DataFrames with a monthly PeriodIndex; weights are DataFrames
indexed by integer year (any scale, e.g. per mille - they are normalised per year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dec(year: int) -> pd.Period:
    return pd.Period(year=year, month=12, freq="M")


def lag(obj: pd.Series | pd.DataFrame, k: int) -> pd.Series | pd.DataFrame:
    """Value k months earlier, aligned on the calendar (robust to gaps in the index)."""
    out = obj.reindex(obj.index - k)
    out.index = obj.index
    return out


def yoy(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Annual rate of change in %."""
    return 100.0 * (obj / lag(obj, 12) - 1.0)


def mom_log(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Month-on-month log change x 100."""
    return 100.0 * np.log(obj / lag(obj, 1))


def normalise(weights: pd.DataFrame) -> pd.DataFrame:
    w = weights.astype(float).fillna(0.0)
    return w.div(w.sum(axis=1), axis=0)


def chain_link(indices: pd.DataFrame, weights: pd.DataFrame,
               rebase_year: int | None = 2025) -> pd.Series:
    """Aggregate component indices with annual chain-linking (December overlap).

    Components with zero or missing weight in a year are ignored that year (e.g. a
    country before it joined the euro area). The result stops at the last month where
    every active component is observed. With `rebase_year`, the output averages 100
    over that year (when the year is complete); otherwise it starts at 100.
    """
    w = normalise(weights.reindex(columns=indices.columns))
    first = next((y for y in sorted(w.index) if dec(y - 1) in indices.index), None)
    if first is None:
        raise ValueError("No year whose previous December is in the index data")

    out: dict[pd.Period, float] = {dec(first - 1): 100.0}
    for y in range(first, int(w.index.max()) + 1):
        link = dec(y - 1)
        if y not in w.index or link not in out:
            break
        wy = w.loc[y]
        active = wy.index[wy > 0]
        months = indices.index[indices.index.year == y]
        if months.empty:
            break
        rel = indices.loc[months, active].div(indices.loc[link, active])
        complete = rel.notna().all(axis=1)
        if not complete.all():
            # keep the leading run of complete months only
            months = months[: int(np.argmin(complete.to_numpy()))]
            rel = rel.loc[months]
        if months.empty:
            break
        agg = out[link] * (rel.to_numpy() @ wy[active].to_numpy())
        out.update(zip(months, agg))
        if len(months) < 12:
            break

    series = pd.Series(out).sort_index()
    series.index = pd.PeriodIndex(series.index, freq="M")
    if rebase_year is not None:
        base = series[series.index.year == rebase_year]
        if len(base) == 12:
            series = 100.0 * series / base.mean()
    return series


def contributions(indices: pd.DataFrame, weights: pd.DataFrame,
                  aggregate: pd.Series | None = None,
                  periods: pd.PeriodIndex | None = None) -> pd.DataFrame:
    """Exact contributions (pp) of each component to the aggregate's annual rate.

    Rows sum to `yoy(aggregate)`. Pass `aggregate` only if it was built from these
    indices and weights (default: rebuilt with `chain_link`).
    """
    w = normalise(weights.reindex(columns=indices.columns))
    agg = aggregate if aggregate is not None else chain_link(indices, weights, rebase_year=None)
    I = indices.to_numpy(dtype=float)
    pos = {p: k for k, p in enumerate(indices.index)}
    P = agg.to_dict()

    rows: dict[pd.Period, np.ndarray] = {}
    for t in (periods if periods is not None else agg.index):
        y = t.year
        D, Dp, t12 = dec(y - 1), dec(y - 2), t - 12
        if t not in P or t12 not in P or D not in P or y not in w.index or (y - 1) not in w.index:
            continue
        wy, wy1 = w.loc[y].to_numpy(), w.loc[y - 1].to_numpy()
        It, ID, It12 = I[pos[t]], I[pos[D]], I[pos[t12]]
        with np.errstate(invalid="ignore", divide="ignore"):
            term1 = (P[D] / P[t12]) * wy * (It / ID - 1.0)
            if t.month == 12:  # t-12 is the link month itself: no second term
                term2 = np.zeros_like(term1)
            else:
                if Dp not in P:
                    continue
                IDp = I[pos[Dp]]
                term2 = (P[Dp] / P[t12]) * wy1 * (ID - It12) / IDp
        term1 = np.where(wy > 0, term1, 0.0)
        term2 = np.where(wy1 > 0, term2, 0.0)
        rows[t] = 100.0 * (term1 + term2)

    out = pd.DataFrame.from_dict(rows, orient="index", columns=indices.columns)
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out.sort_index()


def effective_weights(item_weights: pd.DataFrame, country_weights: pd.DataFrame,
                      countries: list[str], components: list[str]) -> pd.DataFrame:
    """omega_c,i^y: weight of component i of country c in the euro area aggregate.

    item_weights    : year x (geo, coicop), per mille within each country
    country_weights : year x geo, per mille of the euro area
    Returns year x (country, component), each row summing to 1.
    """
    cw = normalise(country_weights.reindex(columns=countries))
    blocks = {}
    for c in countries:
        iw = normalise(item_weights.reindex(columns=pd.MultiIndex.from_product([[c], components])))
        blocks[c] = iw.mul(cw[c].reindex(iw.index).fillna(0.0), axis=0).droplevel(0, axis=1)
    omega = pd.concat(blocks, axis=1).fillna(0.0)
    omega.columns.names = ["geo", "coicop"]
    return omega.loc[omega.sum(axis=1) > 0]


def price_update(weights_prev: pd.Series, dec_now: pd.Series, dec_before: pd.Series) -> pd.Series:
    """Weights for the next year if expenditure shares are unchanged: last year's weights
    price-updated from Dec(y-1) to Dec(y). Used when the official weights are unknown."""
    w = weights_prev * dec_now / dec_before
    return w / w.sum()
