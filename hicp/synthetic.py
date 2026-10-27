"""Synthetic HICP-like data with the same structure as the Eurostat download.

For running the tests and the notebook offline. The numbers are made up - never present
them as euro area data. Prices are generated at country x COICOP-division level; the main
aggregates, countries and the euro area are then built with the real index rules
(evolving composition, annual weights, December link), so both partitions of the headline
hold exactly up to publication rounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregation import chain_link, effective_weights
from .config import Config
from .data import HICPData

ENTRY = {"AT": 1999, "BE": 1999, "DE": 1999, "ES": 1999, "FI": 1999, "FR": 1999, "IE": 1999, "IT": 1999,
         "LU": 1999, "NL": 1999, "PT": 1999, "EL": 2001, "SI": 2007, "CY": 2008, "MT": 2008, "SK": 2009,
         "EE": 2011, "LV": 2014, "LT": 2015, "HR": 2023, "BG": 2026}
SIZE = {"DE": 27, "FR": 20, "IT": 17, "ES": 12, "NL": 6, "BE": 3.7, "AT": 3.3, "IE": 2.4, "PT": 2.1,
        "EL": 2.0, "FI": 1.9, "SK": 0.8, "HR": 0.7, "BG": 0.6, "SI": 0.4, "LT": 0.5, "LU": 0.3,
        "LV": 0.25, "EE": 0.2, "CY": 0.2, "MT": 0.1}
SHARE = {"SERV": 0.46, "NRG": 0.09, "FOOD_P": 0.14, "FOOD_NP": 0.05, "IGD_NNRG": 0.26}
# each synthetic division belongs to one main aggregate (a simplification of the real mapping)
DIVISION_OF = {"CP01": "FOOD_NP", "CP02": "FOOD_P", "CP03": "IGD_NNRG", "CP04": "NRG", "CP05": "IGD_NNRG",
               "CP06": "SERV", "CP07": "NRG", "CP08": "SERV", "CP09": "IGD_NNRG", "CP10": "SERV",
               "CP11": "SERV", "CP12": "SERV", "CP13": "SERV"}
# calendar-month patterns of month-on-month changes (%), mean zero
SEASONAL = {
    "SERV": [-0.6, 0.2, 0.3, 0.3, 0.0, 0.5, 1.0, 0.4, -1.0, -0.2, -0.5, 0.6],
    "IGD_NNRG": [-2.6, -0.2, 2.4, 0.9, 0.1, -0.4, -2.4, -0.3, 2.2, 0.7, 0.0, -0.4],
    "FOOD_P": [0.3, 0.2, 0.0, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, 0.0, -0.1, 0.0],
    "FOOD_NP": [0.6, 0.4, 0.2, 0.2, 0.3, -0.2, -0.8, -0.6, -0.3, 0.1, 0.0, 0.1],
    "NRG": [0.2, 0.1, 0.1, 0.1, 0.0, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0],
}


def _window(periods: pd.PeriodIndex, start: str, end: str) -> np.ndarray:
    return (periods >= pd.Period(start, freq="M")) & (periods <= pd.Period(end, freq="M"))


def _energy_common(periods: pd.PeriodIndex, rng: np.random.Generator) -> np.ndarray:
    """Common energy price changes: oil swings, the 2021-22 gas spike and its reversal."""
    x = rng.normal(0, 0.9, len(periods))
    for start, end, monthly in (("2007-06", "2008-07", 0.9), ("2008-08", "2009-02", -2.5), ("2014-07", "2015-01", -1.8),
                                ("2020-02", "2020-04", -3.0), ("2021-01", "2022-10", 1.6), ("2022-11", "2023-06", -1.4)):
        x[_window(periods, start, end)] += monthly
    x[0] = 0.0
    return x


def make_demo_data(cfg: Config, seed: int = 7, last: str = "2026-09") -> HICPData:
    rng = np.random.default_rng(seed)
    countries, comps, area, codes = list(cfg["countries"]), cfg.components, cfg["area"], cfg["codes"]
    divisions = cfg.divisions or comps
    group = {d: DIVISION_OF.get(d, d) for d in divisions}
    periods = pd.period_range("1998-12", last, freq="M")
    years = list(range(1999, periods[-1].year + 1))
    month = periods.month.to_numpy() - 1

    # weights (per mille): country weights, then division weights inside each country
    cw = pd.DataFrame(index=years, columns=countries, dtype=float)
    for c in countries:
        drift = np.cumsum(rng.normal(0, 0.01, len(years)))
        cw[c] = [SIZE.get(c, 0.5) * np.exp(d) if y >= ENTRY.get(c, 1999) else 0.0 for y, d in zip(years, drift)]
    cw = (1000 * cw.div(cw.sum(axis=1), axis=0)).round(2)
    cw.index.name, cw.columns.name = "year", "geo"

    n_in = pd.Series(group).value_counts()
    iw = {}
    for c in countries:
        base = np.array([SHARE.get(group[d], 0.1) / n_in[group[d]] for d in divisions]) \
            * np.exp(rng.normal(0, 0.12, len(divisions)))
        rows = [base * np.exp(rng.normal(0, 0.02, len(divisions))) for _ in years]
        w = pd.DataFrame(np.round([1000 * r / r.sum() for r in rows], 2), index=years, columns=divisions)
        w.iloc[:, 0] += 1000 - w.sum(axis=1)  # rounded weights sum to exactly 1000
        iw[c] = w

    # monthly price changes (%) per country x division
    surge = np.where(_window(periods, "2021-07", "2023-06"), 0.35, 0.0)
    trend = 0.13 + surge
    common_nrg = _energy_common(periods, rng)
    flat = {}
    for c in countries:
        nrg = 0.8 * common_nrg + rng.normal(0, 0.8, len(periods)) + np.array(SEASONAL["NRG"])[month]
        lagged = pd.Series(nrg).rolling(12, min_periods=1).mean().shift(1).fillna(0.0).to_numpy()
        by_comp = {
            "NRG": nrg,
            "FOOD_P": trend + 0.09 * lagged + np.array(SEASONAL["FOOD_P"])[month],
            "FOOD_NP": trend + 0.05 * lagged + np.array(SEASONAL["FOOD_NP"])[month],
            "IGD_NNRG": 0.6 * trend - 0.05 + 0.05 * lagged + np.array(SEASONAL["IGD_NNRG"])[month],
            "SERV": trend + 0.06 + 0.02 * lagged + np.array(SEASONAL["SERV"])[month],
        }
        for d in divisions:
            x = by_comp.get(group[d], trend) + rng.normal(0, 0.3, len(periods))
            x[0] = 0.0
            flat[(c, d)] = 100 * np.exp(np.cumsum(x) / 100)
    flat = pd.DataFrame(flat, index=periods)
    flat.columns = pd.MultiIndex.from_tuples(flat.columns, names=["geo", "coicop"])
    item_div = pd.concat(iw, axis=1)
    item_div.columns.names = ["geo", "coicop"]

    # main aggregates per country = sums of their divisions' weights
    item_comp = pd.concat({c: pd.DataFrame({k: iw[c][[d for d in divisions if group[d] == k]].sum(axis=1)
                                            for k in comps}) for c in countries}, axis=1)
    item_comp.columns.names = ["geo", "coicop"]

    # aggregates, exactly as Eurostat builds them
    omega = effective_weights(item_div, cw, countries, divisions)
    in_comp = lambda k: [d for d in divisions if group[d] == k]  # noqa: E731
    series = {}
    for c in countries:
        series[(c, codes["headline"])] = chain_link(flat[c], iw[c])
        for k in comps:
            series[(c, k)] = chain_link(flat[c][in_comp(k)], iw[c][in_comp(k)])
        if cfg.divisions:
            for d in divisions:
                series[(c, d)] = flat[(c, d)]
    for d in divisions if cfg.divisions else []:
        series[(area, d)] = chain_link(flat.xs(d, axis=1, level="coicop"), omega.xs(d, axis=1, level="coicop"))
    for k in comps:
        cols = [col for col in flat.columns if group[col[1]] == k]
        series[(area, k)] = chain_link(flat[cols], omega[cols])
    series[(area, codes["headline"])] = chain_link(flat, omega)
    core_cols = [col for col in flat.columns if group[col[1]] in ("SERV", "IGD_NNRG")]
    if codes.get("core"):
        series[(area, codes["core"])] = chain_link(flat[core_cols], omega[core_cols])

    indices = pd.DataFrame(series)
    indices = (100 * indices / indices[indices.index.year == 2025].mean()).round(2).loc["1999-12":]
    indices.columns = pd.MultiIndex.from_tuples(indices.columns, names=["geo", "coicop"])

    ea_div = 1000 * omega.T.groupby(level="coicop").sum().T
    ea_comp = pd.DataFrame({k: ea_div[in_comp(k)].sum(axis=1) for k in comps})
    parts = [item_comp, pd.concat({area: ea_comp.round(2)}, axis=1)]
    if cfg.divisions:
        parts += [item_div, pd.concat({area: ea_div.round(2)}, axis=1)]
    item_w = pd.concat(parts, axis=1)
    item_w.index.name = "year"
    item_w.columns.names = ["geo", "coicop"]

    labels = {codes["headline"]: "All-items HICP", codes.get("core", "core"): "Core (synthetic)",
              **cfg["components"], **cfg.get("divisions", {})}
    present = {"headline": codes["headline"], **({"core": codes["core"]} if codes.get("core") else {})}
    return HICPData(indices=indices, item_weights=item_w, country_weights=cw, area=area, countries=countries,
                    components=comps, divisions=cfg.divisions, codes=present, labels=labels,
                    vintage="SYNTHETIC DEMO DATA", source="synthetic", cleaning={"synthetic data": 0})
