"""Download (Eurostat API), clean, store and load the HICP panels used everywhere else.

In memory (`HICPData`):
    indices          monthly PeriodIndex x (geo, coicop)   index, 2025 = 100
    item_weights     year x (geo, coicop)                  per mille of each country's basket
    country_weights  year x geo                            per mille of the euro area
    flags            long table of Eurostat status flags (p provisional, e estimated, ...)
    quality          one row per series: coverage, gaps, values removed, flags
On disk: tidy parquet files in data/processed/ plus metadata.json (vintage, cleaning counts).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import eurostat, quality
from .config import Config


@dataclass
class HICPData:
    indices: pd.DataFrame
    item_weights: pd.DataFrame
    country_weights: pd.DataFrame
    area: str
    countries: list[str]
    components: list[str]
    divisions: list[str]
    codes: dict[str, str]
    labels: dict[str, str] = field(default_factory=dict)
    vintage: str = ""
    source: str = "Eurostat API"
    flags: pd.DataFrame | None = None
    quality: pd.DataFrame | None = None
    cleaning: dict = field(default_factory=dict)

    @property
    def last_month(self) -> pd.Period:
        return self.headline().index[-1]

    def has(self, geo: str, code: str) -> bool:
        return (geo, code) in self.indices.columns and self.indices[(geo, code)].notna().any()

    def series(self, geo: str, code: str) -> pd.Series:
        return self.indices[(geo, code)].dropna()

    def headline(self, geo: str | None = None) -> pd.Series:
        return self.series(geo or self.area, self.codes["headline"])

    # a "breakdown" is a partition of the headline: main aggregates or COICOP divisions
    def area_breakdown(self, codes: list[str] | None = None) -> pd.DataFrame:
        """Euro area indices of a breakdown (months x codes)."""
        return self.indices[self.area].reindex(columns=codes or self.components)

    def area_weights(self, codes: list[str] | None = None) -> pd.DataFrame:
        """Euro area weights of a breakdown (years x codes, per mille)."""
        return self.item_weights[self.area].reindex(columns=codes or self.components)

    def country_breakdown(self, codes: list[str] | None = None) -> pd.DataFrame:
        """Country x code indices with (geo, coicop) columns."""
        cols = pd.MultiIndex.from_product([self.countries, codes or self.components], names=["geo", "coicop"])
        return self.indices.reindex(columns=cols)

    def area_components(self) -> pd.DataFrame:
        return self.area_breakdown(self.components)

    def area_component_weights(self) -> pd.DataFrame:
        return self.area_weights(self.components)

    def country_components(self) -> pd.DataFrame:
        return self.country_breakdown(self.components)

    def label(self, code: str) -> str:
        return self.labels.get(code, code)


# ---- download ---------------------------------------------------------------------

def _filters(dims: dict, wanted: dict) -> dict:
    """Keep only filters on dimensions the dataset actually has."""
    return {k: v for k, v in wanted.items() if k in dims}


def _single_valued(df: pd.DataFrame, keep: list[str], dataset: str) -> pd.DataFrame:
    """Fail loudly if a dimension we do not filter on still has several values."""
    for dim in df.columns:
        if dim in keep or dim in ("value", "flag"):
            continue
        values = df[dim].unique()
        if len(values) > 1:
            raise eurostat.EurostatError(f"{dataset}: dimension '{dim}' has several values {list(values)[:10]}; "
                                         "add a filter for it in config.yaml")
    return df


def download(cfg: Config, refresh: bool = False) -> HICPData:
    """Pull indices, item weights and country weights from the Eurostat API, then clean."""
    es = cfg["eurostat"]
    ds, base, cache = es["datasets"], es["base_url"], cfg.raw_dir

    meta = eurostat.describe(ds["index"], geo=cfg["area"], cache_dir=cache, base_url=base)
    cdim = eurostat.coicop_dimension(meta)
    available = meta[cdim]
    if missing := [c for c in cfg.required_codes if c not in available]:
        raise eurostat.EurostatError(f"Codes {missing} not in {ds['index']}; fix config.yaml "
                                     f"(`python -m hicp.eurostat describe {ds['index']} --search <word>`)")
    optional = [c for c in cfg.optional_codes if c in available]
    for c in set(cfg.optional_codes) - set(optional):
        warnings.warn(f"Optional code {c} not in {ds['index']}; skipped")
    codes = cfg.required_codes + optional

    frames = []
    for geo in cfg.geos:
        js = eurostat.get_jsonstat(
            ds["index"], _filters(meta, {"freq": "M", "unit": es["index_unit"], "geo": geo, cdim: codes}),
            since=es.get("since"), cache_dir=cache, refresh=refresh, base_url=base)
        frames.append(eurostat.jsonstat_to_frame(js))
    idx = _single_valued(pd.concat(frames, ignore_index=True), ["geo", cdim, "time"], ds["index"])
    idx = idx.rename(columns={cdim: "coicop"})
    idx["time"] = eurostat.to_period(idx["time"])

    meta_w = eurostat.describe(ds["item_weights"], geo=cfg["area"], cache_dir=cache, base_url=base)
    cdim_w = eurostat.coicop_dimension(meta_w)
    if missing := [c for c in cfg.required_codes if c not in meta_w[cdim_w]]:
        raise eurostat.EurostatError(f"Codes {missing} have no weights in {ds['item_weights']}")
    wcodes = [c for c in codes if c in meta_w[cdim_w]]
    frames = []
    for geo in cfg.geos:
        js = eurostat.get_jsonstat(
            ds["item_weights"],
            _filters(meta_w, {"freq": "A", **es.get("item_weight_filter", {}), "geo": geo, cdim_w: wcodes}),
            cache_dir=cache, refresh=refresh, base_url=base)
        frames.append(eurostat.jsonstat_to_frame(js))
    iw = _single_valued(pd.concat(frames, ignore_index=True), ["geo", cdim_w, "time"], ds["item_weights"])
    iw = iw.rename(columns={cdim_w: "coicop"})

    meta_c = eurostat.describe(ds["country_weights"], geo=cfg["countries"][0], cache_dir=cache, base_url=base)
    js = eurostat.get_jsonstat(ds["country_weights"],
                               _filters(meta_c, {"freq": "A", **es.get("country_weight_filter", {})}),
                               cache_dir=cache, refresh=refresh, base_url=base)
    cw = _single_valued(eurostat.jsonstat_to_frame(js), ["geo", "time"], ds["country_weights"])
    cw = cw[cw["geo"].isin(cfg["countries"])]

    # ---- cleaning ----
    idx, idx_log = quality.clean_indices(idx)
    iw, iw_log = quality.clean_weights(iw, ["time", "geo", "coicop"], "item weights")
    cw, cw_log = quality.clean_weights(cw, ["time", "geo"], "country weights")
    flags = idx.loc[idx["flag"].notna() & (idx["flag"] != ""), ["time", "geo", "coicop", "flag"]].reset_index(drop=True)

    indices = idx.pivot(index="time", columns=["geo", "coicop"], values="value").sort_index()
    indices.index = pd.PeriodIndex(indices.index, freq="M")
    labels = {k: v for k, v in available.items() if k in codes}
    labels.update(cfg["components"])
    labels.update(cfg.get("divisions", {}))
    names = {v: k for k, v in cfg["codes"].items()}
    return HICPData(
        indices=indices, item_weights=_annual(iw, ["geo", "coicop"]), country_weights=_annual(cw, ["geo"]),
        area=cfg["area"], countries=list(cfg["countries"]), components=cfg.components, divisions=cfg.divisions,
        codes={"headline": cfg["codes"]["headline"], **{names[c]: c for c in optional}}, labels=labels,
        vintage=datetime.now(timezone.utc).isoformat(timespec="seconds"), flags=flags,
        quality=quality.series_report(indices, flags, idx_log["removed_rows"]),
        cleaning={"index exact duplicates dropped": idx_log["exact duplicates dropped"],
                  "index values removed (non-positive or non-finite)": idx_log["non-positive or non-finite removed"],
                  "item-weight exact duplicates dropped": iw_log["exact duplicates dropped"],
                  "country-weight exact duplicates dropped": cw_log["exact duplicates dropped"],
                  "observations with a status flag": int(len(flags))})


def _annual(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.assign(time=df["time"].astype(str).str[:4].astype(int))
    wide = df.pivot(index="time", columns=columns, values="value").sort_index()
    wide.index.name = "year"
    return wide


# ---- storage ----------------------------------------------------------------------

def _long(wide: pd.DataFrame, index_name: str) -> pd.DataFrame:
    return wide.rename_axis(index=index_name).melt(ignore_index=False).dropna(subset=["value"]).reset_index()


def save(data: HICPData, cfg: Config) -> None:
    out = cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    idx = _long(data.indices, "time")
    idx["time"] = idx["time"].dt.to_timestamp()
    idx.to_parquet(out / "indices.parquet", index=False)
    _long(data.item_weights, "year").to_parquet(out / "item_weights.parquet", index=False)
    _long(data.country_weights, "year").to_parquet(out / "country_weights.parquet", index=False)
    if data.flags is not None:
        data.flags.assign(time=data.flags["time"].astype(str)).to_parquet(out / "flags.parquet", index=False)
    if data.quality is not None:
        data.quality.assign(flags=data.quality["flags"].astype(str)).to_csv(out / "quality.csv", index=False)
    meta = {"source": data.source, "vintage": data.vintage, "labels": data.labels, "area": data.area,
            "countries": data.countries, "components": data.components, "divisions": data.divisions,
            "codes": data.codes, "cleaning": data.cleaning}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(cfg: Config) -> HICPData:
    src = cfg.processed_dir
    meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    idx = pd.read_parquet(src / "indices.parquet")
    indices = idx.pivot(index="time", columns=["geo", "coicop"], values="value").sort_index()
    indices.index = pd.PeriodIndex(pd.to_datetime(indices.index), freq="M")
    iw = pd.read_parquet(src / "item_weights.parquet").pivot(index="year", columns=["geo", "coicop"], values="value")
    cw = pd.read_parquet(src / "country_weights.parquet").pivot(index="year", columns="geo", values="value")
    flags = pd.read_parquet(src / "flags.parquet") if (src / "flags.parquet").exists() else None
    if flags is not None:
        flags["time"] = pd.PeriodIndex(flags["time"], freq="M")
    qual = pd.read_csv(src / "quality.csv") if (src / "quality.csv").exists() else None
    return HICPData(indices=indices, item_weights=iw, country_weights=cw, area=meta["area"],
                    countries=meta["countries"], components=meta["components"], divisions=meta.get("divisions", []),
                    codes=meta["codes"], labels=meta["labels"], vintage=meta["vintage"],
                    source=meta.get("source", ""), flags=flags, quality=qual, cleaning=meta.get("cleaning", {}))


def load_or_download(cfg: Config, refresh: bool = False) -> HICPData:
    """Use data/processed/ if present (and not refreshing), else download, clean and save."""
    if not refresh and (cfg.processed_dir / "metadata.json").exists():
        return load(cfg)
    data = download(cfg, refresh=refresh)
    save(data, cfg)
    return data
