"""External market data used to size the oil-shock preset (FRED, no API key), plus the
shares of fuels, electricity and gas in energy from Eurostat's own weights."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from .data import HICPData

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fred_series(series: str, cache_dir: Path | None = None, refresh: bool = False) -> pd.Series:
    """Daily FRED series (missing values dropped), cached as CSV."""
    path = Path(cache_dir) / f"fred_{series}.csv" if cache_dir else None
    if path is not None and path.exists() and not refresh:
        text = path.read_text(encoding="utf-8")
    else:
        resp = requests.get(FRED_CSV.format(series=series), timeout=60)
        resp.raise_for_status()
        text = resp.text
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    df = pd.read_csv(io.StringIO(text), na_values=".")
    s = df.set_index(pd.to_datetime(df.iloc[:, 0]))[series].astype(float).dropna()
    return s.rename(series)


def fuels_share_of_energy(data: HICPData, fallback: float) -> tuple[float, str]:
    """Weight of liquid fuels and transport fuels in the euro area energy weight, latest year."""
    fuels, energy = data.codes.get("fuels"), "NRG" if "NRG" in data.components else None
    if fuels and energy and (data.area, fuels) in data.item_weights.columns:
        w = data.item_weights[data.area][[fuels, energy]].dropna()
        if len(w):
            y = int(w.index[-1])
            return float(w.loc[y, fuels] / w.loc[y, energy]), f"Eurostat item weights {y} ({fuels} / {energy})"
    return fallback, "config.yaml fallback"


def share_of_energy(data: HICPData, codes: list[str], geo: str, fallback: float,
                    energy: str = "NRG") -> tuple[float, str]:
    """Weight of `codes` (e.g. electricity and gas) in the energy weight of `geo`, latest year."""
    cols = [(geo, c) for c in (*codes, energy)]
    if codes and all(c in data.item_weights.columns for c in cols):
        w = data.item_weights[cols].dropna()
        if len(w):
            y = int(w.index[-1])
            return float(w.loc[y, cols[:-1]].sum() / w.loc[y, cols[-1]]), f"Eurostat item weights {y}"
    return fallback, "config.yaml fallback"


def market_snapshot(cfg, data: HICPData, refresh: bool = False) -> dict:
    """Latest Brent (USD/bbl) and USD per EUR, 30-day averages, with their dates."""
    m = cfg["market"]
    brent = fred_series(m["brent"], cfg.raw_dir, refresh)
    fx = fred_series(m["usd_per_eur"], cfg.raw_dir, refresh)
    share, share_src = fuels_share_of_energy(data, cfg["shocks"]["fuels_share_fallback"])
    last = min(brent.index[-1], fx.index[-1])
    window = lambda s: s[(s.index > last - pd.Timedelta(days=30)) & (s.index <= last)]  # noqa: E731
    return {"brent_usd": round(float(window(brent).mean()), 2), "usd_per_eur": round(float(window(fx).mean()), 4),
            "as_of": str(last.date()), "source": f"FRED {m['brent']}, {m['usd_per_eur']} (30-day averages)",
            "fuels_share_of_energy": round(share, 4), "fuels_share_source": share_src}
