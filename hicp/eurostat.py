"""Thin client for the Eurostat dissemination API (JSON-stat 2.0).

Every response is cached in data/raw/ together with the time it was downloaded (the
"vintage"), so a run can be reproduced exactly; pass ``refresh=True`` to download again.

Command line helper to inspect a dataset before configuring the pipeline:

    python -m hicp.eurostat describe prc_hicp_minr
    python -m hicp.eurostat describe prc_hicp_minr --search energy
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


class EurostatError(RuntimeError):
    """The API answered with an error message (bad dataset code, filter, ...)."""


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _params(filters: dict[str, Any], since: str | None, until: str | None,
            last: int | None) -> list[tuple[str, str]]:
    params = [("format", "JSON"), ("lang", "EN")]
    for dim, value in filters.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        params += [(dim, str(v)) for v in values]
    if since:
        params.append(("sinceTimePeriod", since))
    if until:
        params.append(("untilTimePeriod", until))
    if last:
        params.append(("lastTimePeriod", str(last)))
    return params


def _raise_api_error(payload: dict, status: int) -> None:
    err = payload.get("error")
    if not err:
        return
    items = err if isinstance(err, list) else [err]
    msg = "; ".join(str(e.get("label", e)) for e in items)
    raise EurostatError(f"Eurostat API error (HTTP {status}): {msg}")


def get_jsonstat(dataset: str, filters: dict[str, Any] | None = None, *,
                 since: str | None = None, until: str | None = None, last: int | None = None,
                 cache_dir: Path | None = None, refresh: bool = False,
                 base_url: str = BASE_URL, timeout: int = 180) -> dict:
    """Download one dataset slice as JSON-stat, using the on-disk cache when available."""
    params = _params(filters or {}, since, until, last)
    key = hashlib.sha1(json.dumps([dataset, params]).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{dataset}_{key}.json" if cache_dir else None
    if path is not None and path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    resp = _session().get(f"{base_url}/{dataset}", params=params, timeout=timeout)
    try:
        payload = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise EurostatError(f"Non-JSON answer for {resp.url}")
    _raise_api_error(payload, resp.status_code)
    resp.raise_for_status()

    payload["_retrieved"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["_request"] = {"dataset": dataset, "params": params, "url": resp.url}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _category_codes(dimension: dict) -> list[str]:
    index = dimension["category"]["index"]
    if isinstance(index, dict):
        return sorted(index, key=index.get)
    return list(index)


def jsonstat_to_frame(js: dict) -> pd.DataFrame:
    """Flatten a JSON-stat 2.0 dataset into a long frame: one column per dimension,
    plus `value` and `flag` (Eurostat status: p = provisional, e = estimated, ...)."""
    dims: list[str] = js["id"]
    sizes: list[int] = js["size"]
    codes = [_category_codes(js["dimension"][d]) for d in dims]

    values = js.get("value", {})
    if isinstance(values, dict):
        pos = np.fromiter((int(k) for k in values), dtype=np.int64, count=len(values))
        vals = np.array([np.nan if v is None else v for v in values.values()], dtype=float)
    else:
        arr = np.array([np.nan if v is None else v for v in values], dtype=float)
        pos = np.flatnonzero(~np.isnan(arr))
        vals = arr[pos]

    status = js.get("status") or {}
    if isinstance(status, list):
        flags = [status[p] if p < len(status) else None for p in pos]
    else:
        flags = [status.get(str(p)) for p in pos]

    coords = np.unravel_index(pos, sizes) if len(pos) else [np.array([], dtype=int)] * len(dims)
    out = {d: np.asarray(c, dtype=object)[ix] for d, c, ix in zip(dims, codes, coords)}
    out["value"] = vals
    out["flag"] = flags
    return pd.DataFrame(out)


def labels(js: dict, dim: str) -> dict[str, str]:
    return dict(js["dimension"][dim]["category"].get("label", {}))


def describe(dataset: str, geo: str = "EA", *, cache_dir: Path | None = None,
             base_url: str = BASE_URL) -> dict[str, dict[str, str]]:
    """Dimension -> {code: label} for the latest period of `dataset` (a small request)."""
    js = get_jsonstat(dataset, {"geo": geo}, last=1, cache_dir=cache_dir, base_url=base_url)
    return {d: labels(js, d) for d in js["id"]}


def coicop_dimension(dims: Iterable[str]) -> str:
    """Name of the COICOP dimension (its exact name changed with ECOICOP ver.2)."""
    for d in dims:
        if d.lower().startswith("coicop"):
            return d
    raise EurostatError(f"No COICOP dimension among {list(dims)}")


def to_period(time_codes: pd.Series) -> pd.PeriodIndex:
    """'2025-01' or '2025M01' -> monthly Period."""
    return pd.PeriodIndex(time_codes.astype(str).str.replace("M", "-", regex=False), freq="M")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Eurostat dataset")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="list dimensions and codes of a dataset")
    d.add_argument("dataset")
    d.add_argument("--geo", default="EA")
    d.add_argument("--search", default=None, help="only show codes whose label contains this text")
    args = parser.parse_args()

    meta = describe(args.dataset, geo=args.geo)
    for dim, cats in meta.items():
        print(f"\n== {dim} ({len(cats)} codes)")
        for code, label in cats.items():
            if args.search and args.search.lower() not in label.lower():
                continue
            print(f"  {code:<20} {label}")


if __name__ == "__main__":
    _main()
