"""Data cleaning and quality report for the Eurostat extractions.

Rules (nothing is interpolated or invented):
  - exact duplicate observations are dropped; conflicting duplicates (same key, different
    value) stop the pipeline;
  - non-finite or non-positive index values are removed (an index cannot be <= 0);
  - negative weights stop the pipeline; missing weights mean "not in the basket";
  - Eurostat status flags (p provisional, e estimated, u low reliability, ...) are kept
    and reported, not used to drop data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataQualityError(ValueError):
    """The extraction cannot be used as is."""


def _dedupe(df: pd.DataFrame, keys: list[str], what: str) -> tuple[pd.DataFrame, int]:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        conflicts = df[dup].groupby(keys)["value"].nunique()
        if (conflicts > 1).any():
            raise DataQualityError(f"{what}: conflicting duplicates for {list(conflicts[conflicts > 1].index[:5])}")
    n = int(df.duplicated(keys).sum())
    return df.drop_duplicates(keys), n


def clean_indices(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Long frame (time, geo, coicop, value, flag) -> cleaned frame + counts."""
    df, n_dup = _dedupe(df, ["time", "geo", "coicop"], "indices")
    bad = ~np.isfinite(df["value"].astype(float)) | (df["value"] <= 0)
    removed = df[bad]
    return df[~bad].copy(), {"exact duplicates dropped": n_dup, "non-positive or non-finite removed": int(bad.sum()),
                             "removed_rows": removed[["time", "geo", "coicop", "value"]]}


def clean_weights(df: pd.DataFrame, keys: list[str], what: str) -> tuple[pd.DataFrame, dict]:
    df, n_dup = _dedupe(df.dropna(subset=["value"]), keys, what)
    if (df["value"] < 0).any():
        raise DataQualityError(f"{what}: negative weights {df[df['value'] < 0].head().to_dict('records')}")
    return df, {"exact duplicates dropped": n_dup}


def series_report(indices: pd.DataFrame, flags: pd.DataFrame | None, removed: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (geo, coicop): coverage, interior gaps, values removed, flagged months."""
    rows = []
    flag_counts = (flags.groupby(["geo", "coicop"])["flag"].agg(lambda f: dict(f.value_counts()))
                   if flags is not None and len(flags) else pd.Series(dtype=object))
    last_flag = (flags.sort_values("time").groupby(["geo", "coicop"]).last()
                 if flags is not None and len(flags) else pd.DataFrame())
    removed_counts = removed.groupby(["geo", "coicop"]).size() if removed is not None and len(removed) else pd.Series(dtype=int)
    for col in indices.columns:
        s = indices[col].dropna()
        if s.empty:
            rows.append({"geo": col[0], "coicop": col[1], "observations": 0})
            continue
        span = pd.period_range(s.index[0], s.index[-1], freq="M")
        row = {"geo": col[0], "coicop": col[1], "first": str(s.index[0]), "last": str(s.index[-1]),
               "observations": len(s), "interior gaps": len(span) - len(s),
               "values removed": int(removed_counts.get(col, 0)), "flags": flag_counts.get(col, {})}
        if col in last_flag.index:
            row["latest flagged month"] = f"{last_flag.loc[col, 'time']} ({last_flag.loc[col, 'flag']})"
        rows.append(row)
    return pd.DataFrame(rows)
