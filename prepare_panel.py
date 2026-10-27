"""Prépare et audite le panel, pays par pays, depuis une collecte complète.

Utilisation : uv run python prepare_panel.py [--manifest chemin/du/manifeste.json]
Les observations et les indicateurs de qualité sont conservés sans imputation.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
import prepare_data
from download_panel import build_jobs
from prepare_data import jsonstat_to_frame

WEIGHT_TOLERANCE = 0.2  # Sur l'échelle publiée : total = 1000.
DIVISIONS = [f"CP{number:02d}" for number in range(1, 14)]


def select_manifest(explicit: Path | None = None) -> tuple[Path, dict]:
    expected = {job["name"]: job for job in build_jobs()}
    candidates = (
        [explicit]
        if explicit is not None
        else sorted(
            (config.RAW_DATA_DIR / "collections").glob("panel_*.json"),
            reverse=True,
        )
    )
    signature = {
        "start_period": config.START_PERIOD,
        "end_period": config.END_PERIOD,
        "classification": config.CLASSIFICATION,
        "index_unit": config.INDEX_UNIT,
        "euro_area_entry": config.EURO_AREA_ENTRY,
    }
    for path in candidates:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        jobs = manifest.get("jobs", [])
        compatible = (
            manifest.get("status") == "complete"
            and all(manifest.get(key) == value for key, value in signature.items())
            and len(jobs) == len(expected)
            and {job["name"] for job in jobs} == set(expected)
            and all(
                job.get("status") == "complete"
                and job["dataset"] == expected[job["name"]]["dataset"]
                and job["filters"] == expected[job["name"]]["filters"]
                for job in jobs
            )
        )
        if compatible:
            return path, manifest
        if explicit is not None:
            raise ValueError(
                "Ce manifeste n'est pas complet ou compatible avec config.py."
            )
    raise FileNotFoundError(
        "Aucune collecte complète compatible. Exécuter download_panel.py."
    )


def read_job(job: dict) -> tuple[pd.DataFrame, list[str], dict]:
    source = config.PROJECT_ROOT / job["raw_file"]
    metadata = json.loads(
        (config.PROJECT_ROOT / job["metadata_file"]).read_text(encoding="utf-8")
    )
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != job["sha256"] or digest != metadata["sha256"]:
        raise ValueError(f"Empreinte incorrecte : {source}")
    if metadata["parameters"] != {"format": "JSON", "lang": "EN", **job["filters"]}:
        raise ValueError(f"Paramètres incohérents : {source}")

    payload = json.loads(content)
    frame = jsonstat_to_frame(payload)
    for key in ("freq", "geo", "unit", "statinfo"):
        if key in job["filters"] and not frame[key].eq(job["filters"][key]).all():
            raise ValueError(f"Sélection incorrecte pour {key} : {source}")

    monthly = job["dataset"] == config.DATASETS["indices_and_rates"]
    if monthly:
        frame["date"] = pd.to_datetime(frame["time"], format="%Y-%m")
        frame["year"] = frame["date"].dt.year
        outside = ~frame["time"].between(config.START_PERIOD, config.END_PERIOD)
    else:
        frame["year"] = pd.to_numeric(frame["time"], errors="raise").astype(int)
        outside = ~frame["year"].between(
            int(config.START_PERIOD[:4]), int(config.END_PERIOD[:4])
        )

    stats = {
        "name": job["name"],
        "rows": len(frame),
        "observed": int(frame["value"].notna().sum()),
        "duplicates": int(frame.duplicated(payload["id"]).sum()),
        "negative_values": int(frame["value"].lt(0).sum()),
        "infinite_values": int(
            np.isinf(frame["value"].to_numpy(dtype=float, na_value=np.nan)).sum()
        ),
        "outside_period": int(outside.sum()),
        "zero_indices": int(frame["value"].eq(0).sum()) if monthly else 0,
        "status_counts": {
            str(code): int(count)
            for code, count in frame["status"].dropna().value_counts().items()
        },
    }
    if stats["observed"] != metadata["observation_count"]:
        raise ValueError(f"Nombre d'observations incohérent : {source}")
    return frame, payload["id"], stats


def coverage(frame: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    keys = ["geo", "coicop18"]
    result = frame.groupby(keys, observed=True)["value"].agg(
        cells_received="size", observed_months="count"
    )
    spans = (
        frame.loc[frame["value"].notna()]
        .groupby(keys, observed=True)["date"]
        .agg(first_observation="min", last_observation="max")
    )
    result = result.join(spans).reset_index()
    first = result["first_observation"]
    last = result["last_observation"]
    span = 12 * (last.dt.year - first.dt.year) + last.dt.month - first.dt.month + 1
    result["expected_months"] = len(months)
    result["missing_total"] = len(months) - result["observed_months"]
    result["missing_inside"] = (
        (span - result["observed_months"])
        .where(result["observed_months"].gt(0), 0)
        .astype("Int64")
    )
    result["missing_before_first"] = (
        12 * (first.dt.year - months[0].year) + first.dt.month - months[0].month
    ).astype("Int64")
    result["missing_after_last"] = (
        12 * (months[-1].year - last.dt.year) + months[-1].month - last.dt.month
    ).astype("Int64")
    result["never_observed"] = result["observed_months"].eq(0)
    return result


def item_weight_sums(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["geo", "year"]
    divisions = frame.loc[frame["coicop18"].isin(DIVISIONS)].groupby(keys)["value"]
    sums = divisions.sum(min_count=1).rename("division_sum").to_frame()
    sums["division_count"] = divisions.count()
    totals = frame.loc[frame["coicop18"].eq("TOTAL"), [*keys, "value"]]
    sums = sums.reset_index().merge(
        totals.rename(columns={"value": "total_weight"}),
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    sums["available"] = sums["total_weight"].notna() | sums["division_count"].gt(0)
    sums["bad_total"] = sums["available"] & (
        sums["total_weight"].isna()
        | sums["total_weight"].sub(1000).abs().gt(WEIGHT_TOLERANCE).fillna(False)
    )
    sums["bad_partition"] = sums["available"] & (
        sums["division_count"].ne(13)
        | sums["division_sum"]
        .sub(sums["total_weight"])
        .abs()
        .gt(WEIGHT_TOLERANCE)
        .fillna(True)
    )
    return sums


def audit_country_weights(frame: pd.DataFrame, countries: list[str], years: range):
    grid = pd.MultiIndex.from_product(
        [countries, years], names=["geo", "year"]
    ).to_frame(index=False)
    national = frame.loc[frame["geo"].isin(countries)]
    national = grid.merge(
        national, on=["geo", "year"], how="left", validate="one_to_one"
    )
    entry_year = national["geo"].map(
        {code: int(date[:4]) for code, date in config.EURO_AREA_ENTRY.items()}
    )
    # La série EA rétrospective utilise ses 11 fondateurs avant 1999.
    national["included_in_ea_reference"] = entry_year.eq(1999) | national["year"].ge(
        entry_year
    )
    active = national["included_in_ea_reference"]
    missing_active = active & (national["value"].isna() | national["value"].le(0))
    positive_inactive = ~active & national["value"].gt(0).fillna(False)
    sums = (
        national.loc[active]
        .groupby("year")["value"]
        .agg(expected_countries="size", observed_countries="count")
    )
    sums["weight_sum"] = national.loc[active].groupby("year")["value"].sum(min_count=1)
    sums["bad_sum"] = sums["expected_countries"].ne(sums["observed_countries"]) | sums[
        "weight_sum"
    ].sub(1000).abs().gt(WEIGHT_TOLERANCE).fillna(True)
    errors = {
        "missing_or_zero_active_country_weights": int(missing_active.sum()),
        "positive_weights_outside_ea_reference": int(positive_inactive.sum()),
        "country_weight_sum_failures": int(sums["bad_sum"].sum()),
    }
    national["weight_share"] = national["value"] / 1000
    return national, sums.reset_index(), errors


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def align_weights(weights: pd.DataFrame, monthly_counts: pd.DataFrame, months):
    """Compare les disponibilités sans remplacer les poids manquants."""
    alignment = weights.rename(columns={"value": "item_weight"}).merge(
        monthly_counts,
        on=["geo", "coicop18", "year"],
        how="outer",
        validate="one_to_one",
    )
    alignment["expected_index_months"] = alignment["year"].map(
        pd.Series(months.year).value_counts().to_dict()
    )
    counts = alignment["index_months"].fillna(0)
    positive = alignment["item_weight"].gt(0).fillna(False)
    alignment["positive_weight_without_index"] = positive & counts.eq(0)
    alignment["positive_weight_partial_index"] = (
        positive & counts.gt(0) & counts.lt(alignment["expected_index_months"])
    )
    alignment["index_without_item_weight"] = (
        counts.gt(0) & alignment["item_weight"].isna()
    )
    return alignment


def main(manifest_path: Path | None = None) -> Path:
    source_manifest, manifest = select_manifest(manifest_path)
    output = config.PROCESSED_DATA_DIR / "panel" / manifest["collection_id"]
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "_SUCCESS"
    marker.unlink(missing_ok=True)
    months = pd.date_range(config.START_PERIOD, config.END_PERIOD, freq="MS")
    years = range(int(config.START_PERIOD[:4]), int(config.END_PERIOD[:4]) + 1)
    countries = sorted(
        code
        for code, entry in config.EURO_AREA_ENTRY.items()
        if entry <= config.END_PERIOD
    )
    report = {
        "integrity_status": "running",
        "collection_id": manifest["collection_id"],
        "source_manifest": str(source_manifest.resolve()),
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "transform_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "converter_sha256": hashlib.sha256(
            Path(prepare_data.__file__).read_bytes()
        ).hexdigest(),
        "pandas_version": pd.__version__,
        "weight_scale": 1000,
        "weight_sum_tolerance": WEIGHT_TOLERANCE,
        "jobs": [],
    }
    errors = Counter()
    coverages, monthly_counts, slim_weights, item_sums = [], [], [], []
    zero_indices = []
    country_frame = None
    print(f"Collecte utilisée : {source_manifest}", flush=True)

    try:
        for number, job in enumerate(manifest["jobs"], start=1):
            print(f"[{number}/{len(manifest['jobs'])}] {job['name']}", flush=True)
            frame, _, stats = read_job(job)
            report["jobs"].append(stats)
            for check in (
                "duplicates",
                "negative_values",
                "infinite_values",
                "outside_period",
            ):
                errors[check] += stats[check]

            if job["dataset"] == config.DATASETS["indices_and_rates"]:
                coverages.append(coverage(frame, months))
                zero_indices.append(
                    frame.loc[
                        frame["value"].eq(0),
                        ["geo", "coicop18", "time", "date", "value", "status"],
                    ].copy()
                )
                monthly_counts.append(
                    frame.groupby(["geo", "coicop18", "year"], observed=True)["value"]
                    .count()
                    .rename("index_months")
                    .reset_index()
                )
                folder = output / "indices"
            elif job["dataset"] == config.DATASETS["item_weights"]:
                item_sums.append(item_weight_sums(frame))
                slim_weights.append(frame[["geo", "coicop18", "year", "value"]].copy())
                frame["weight_share"] = frame["value"] / 1000
                folder = output / "item_weights"
            else:
                country_frame = frame
                continue

            folder.mkdir(exist_ok=True)
            frame.to_parquet(
                folder / f"{job['filters']['geo']}.parquet",
                engine="pyarrow",
                index=False,
            )

        if country_frame is None:
            raise ValueError("Les pondérations des pays sont absentes.")
        coverage_table = pd.concat(coverages, ignore_index=True)
        sums_table = pd.concat(item_sums, ignore_index=True)
        weights_table = pd.concat(slim_weights, ignore_index=True)
        country_weights, country_sums, country_errors = audit_country_weights(
            country_frame, countries, years
        )
        errors.update(country_errors)
        errors["item_weight_total_failures"] = int(sums_table["bad_total"].sum())
        errors["item_weight_partition_failures"] = int(
            sums_table["bad_partition"].sum()
        )

        active_totals = country_weights.loc[
            country_weights["included_in_ea_reference"], ["geo", "year"]
        ].merge(
            sums_table[["geo", "year", "total_weight"]],
            on=["geo", "year"],
            how="left",
            validate="one_to_one",
        )
        errors["active_countries_without_item_weights"] = int(
            active_totals["total_weight"].isna().sum()
        )

        alignment = align_weights(
            weights_table, pd.concat(monthly_counts, ignore_index=True), months
        )

        tables = {
            "coverage.parquet": coverage_table,
            "item_weight_sums.parquet": sums_table,
            "country_weights.parquet": country_weights,
            "country_weight_sums.parquet": country_sums,
            "weight_index_alignment.parquet": alignment,
            "zero_indices.parquet": pd.concat(zero_indices, ignore_index=True),
        }
        for name, table in tables.items():
            table.to_parquet(output / name, engine="pyarrow", index=False)

        report["hard_errors"] = dict(errors)
        report["review_summary"] = {
            "series": len(coverage_table),
            "series_never_observed": int(coverage_table["never_observed"].sum()),
            "series_with_internal_gaps": int(
                coverage_table["missing_inside"].gt(0).sum()
            ),
            "series_with_shorter_history": int(
                (
                    coverage_table["observed_months"].gt(0)
                    & (
                        coverage_table["missing_before_first"].gt(0)
                        | coverage_table["missing_after_last"].gt(0)
                    )
                ).sum()
            ),
            "zero_index_cells": sum(job["zero_indices"] for job in report["jobs"]),
            "positive_weight_without_index_years": int(
                alignment["positive_weight_without_index"].sum()
            ),
            "positive_weight_partial_index_years": int(
                alignment["positive_weight_partial_index"].sum()
            ),
            "index_without_item_weight_years": int(
                alignment["index_without_item_weight"].sum()
            ),
        }
        report["country_weight_entities_excluded"] = sorted(
            set(country_frame["geo"]) - set(countries)
        )
        report["source_versions"] = {
            dataset: sorted(
                {
                    str(job.get("dataset_updated"))
                    for job in manifest["jobs"]
                    if job["dataset"] == dataset
                }
            )
            for dataset in {job["dataset"] for job in manifest["jobs"]}
        }
        report["integrity_status"] = "failed" if any(errors.values()) else "passed"
        write_json(output / "audit.json", report)
        print(
            json.dumps(
                {
                    "hard_errors": dict(errors),
                    "review_summary": report["review_summary"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if any(errors.values()):
            raise ValueError(f"Contrôles bloquants échoués : {output / 'audit.json'}")
        marker.write_text(
            "Preparation completed; review coverage before modelling.\n",
            encoding="utf-8",
        )
        print(f"Intégrité : passed. Résultats : {output}")
        return output

    except (Exception, KeyboardInterrupt) as exc:
        report["integrity_status"] = "failed"
        report["exception"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "audit.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    arguments = parser.parse_args()
    main(arguments.manifest)
