"""Consistency checks on the cleaned data, for both partitions of the headline (main
aggregates and COICOP ver.2 divisions). They double as evidence that the aggregation
engine reproduces Eurostat's published numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregation import chain_link, contributions, dec, effective_weights, yoy
from .data import HICPData


def _row(group: str, check: str, value: float, tolerance: float, unit: str, note: str = "") -> dict:
    return {"group": group, "check": check, "value": value, "tolerance": tolerance, "unit": unit,
            "passed": bool(np.isfinite(value) and abs(value) <= tolerance), "note": note}


def breakdowns(data: HICPData) -> dict[str, list[str]]:
    out = {"Main aggregates": data.components}
    if data.divisions:
        out["COICOP divisions"] = data.divisions
    return out


def run_checks(data: HICPData, since: int = 2015, yoy_tol: float = 0.1) -> pd.DataFrame:
    rows = []
    head = data.headline()
    recent = lambda s: s[s.index.year >= since]  # noqa: E731
    geos = [g for g in [data.area, *data.countries] if g in data.item_weights.columns.get_level_values(0)]

    cw = data.country_weights.reindex(columns=data.countries).sum(axis=1)
    rows.append(_row("Weights", "Country weights sum to 1000 (max |dev|)",
                     float((cw[cw.index >= since] - 1000).abs().max()), 1.0, "per mille"))

    for name, codes in breakdowns(data).items():
        iw = data.item_weights
        sums = pd.DataFrame({g: iw[g].reindex(columns=codes).sum(axis=1, min_count=len(codes)) for g in geos})
        rows.append(_row(name, "Weights sum to 1000 in every country (max |dev|)",
                         float(np.nanmax(np.abs(sums[sums.index >= since].to_numpy() - 1000))), 1.0, "per mille"))

        omega = effective_weights(iw, data.country_weights, data.countries, codes)
        implied = 1000 * omega.T.groupby(level="coicop").sum().T.reindex(columns=codes)
        published = data.area_weights(codes)
        years = implied.index.intersection(published.index)
        diff = (implied.loc[years] - published.loc[years])[lambda d: d.index >= since]
        rows.append(_row(name, "EA weights = sum of country weights x item weights (max |dev|)",
                         float(np.nanmax(np.abs(diff.to_numpy()))), 1.0, "per mille"))

        err = recent((yoy(chain_link(data.area_breakdown(codes), data.area_weights(codes))) - yoy(head)).dropna())
        rows.append(_row(name, "EA headline YoY rebuilt from EA series (max |error|)",
                         float(err.abs().max()), yoy_tol, "pp", f"mean |error| {err.abs().mean():.3f}"))

        # start just before the window: some countries' series begin late
        flat = chain_link(data.country_breakdown(codes).loc[dec(since - 2):], omega)
        err = recent((yoy(flat) - yoy(head)).dropna())
        rows.append(_row(name, "EA headline YoY rebuilt from 21 countries (max |error|)",
                         float(err.abs().max()), yoy_tol, "pp", f"mean |error| {err.abs().mean():.3f}"))

        contrib = contributions(data.area_breakdown(codes), data.area_weights(codes))
        err = recent((contrib.sum(axis=1) - yoy(head)).dropna())
        rows.append(_row(name, "Exact contributions sum to published YoY (max |error|)",
                         float(err.abs().max()), yoy_tol, "pp"))

        needed = pd.concat([data.area_breakdown(codes), data.country_breakdown(codes)], axis=1)
        window = needed[needed.index.year >= since]
        missing = int(window.isna().sum().sum() - window.iloc[-1:].isna().sum().sum())
        rows.append(_row(name, "Missing values since 2015 (excl. latest month)", missing, 0, "cells"))

    for what, n in data.cleaning.items():
        rows.append({"group": "Cleaning", "check": what, "value": n, "tolerance": np.nan, "unit": "count",
                     "passed": True, "note": "informational"})
    return pd.DataFrame(rows)
