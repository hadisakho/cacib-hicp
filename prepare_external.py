"""Télécharge et prépare trois PPI Eurostat, sans imputation.

uv run python prepare_external.py
Décalage prudent de deux mois par rapport au dernier mois IPCH observé.
Composition fixe EA20 : indicateur explicatif distinct de la cible EA évolutive.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
import eurostat_client
import prepare_data
from eurostat_client import download_dataset
from prepare_data import jsonstat_to_frame

DATASET = "sts_inppd_m"
GEO = "EA20"
SECTORS = {
    "ppi_energy": "MIG_NRG",
    "ppi_intermediate": "MIG_ING",
    "ppi_food": "C10-C12",
}
LAG_MONTHS = 2


def make_features(indices, origins, lag_months=LAG_MONTHS):
    """Indexation par mois de référence, puis décalage calendaire explicite."""
    if lag_months < 2:
        raise ValueError("Ce protocole impose au moins deux mois de décalage.")
    if not isinstance(indices.index, pd.PeriodIndex) or indices.index.freqstr != "M":
        raise ValueError("Les indices doivent avoir un PeriodIndex mensuel.")
    if indices.index.has_duplicates or not indices.index.is_monotonic_increasing:
        raise ValueError("Mois dupliqués ou non triés.")
    dense = indices.reindex(
        pd.period_range(indices.index.min(), indices.index.max(), freq="M")
    )
    observed = dense.to_numpy(dtype=float)
    if np.isinf(observed).any() or (observed <= 0).any():
        raise ValueError("Indices PPI non positifs ou infinis.")
    changes = pd.DataFrame(index=dense.index)
    for name in SECTORS:
        changes[name + "_mom"] = 100 * (dense[name] / dense[name].shift(1) - 1)
        changes[name + "_yoy"] = 100 * (dense[name] / dense[name].shift(12) - 1)
    changes.index = changes.index + lag_months
    result = changes.reindex(origins)
    return result


def decode_snapshot(path, filters):
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != metadata["sha256"]:
        raise ValueError("Empreinte incorrecte du téléchargement PPI.")
    if metadata["parameters"] != {"format": "JSON", "lang": "EN", **filters}:
        raise ValueError("Paramètres du téléchargement PPI incohérents.")
    frame = jsonstat_to_frame(json.loads(content))
    for key in ("freq", "indic_bt", "nace_r2", "s_adj", "unit", "geo"):
        if not frame[key].eq(filters[key]).all():
            raise ValueError(f"Dimension PPI inattendue : {key}")
    frame["month"] = pd.PeriodIndex(frame["time"], freq="M")
    if frame["month"].duplicated().any():
        raise ValueError("Plusieurs valeurs PPI pour le même mois.")
    expected = pd.period_range(config.START_PERIOD, config.END_PERIOD, freq="M")
    if not frame["month"].isin(expected).all():
        raise ValueError("Des mois PPI sortent de la période demandée.")
    values = frame["value"].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or (values <= 0).any() or not np.isfinite(values).any():
        raise ValueError("Valeurs PPI absentes, infinies ou non positives.")
    return frame, metadata


def main():
    stamp = datetime.now(UTC).strftime("external_%Y%m%dT%H%M%S_%fZ")
    output = config.PROCESSED_DATA_DIR / "external" / stamp
    output.mkdir(parents=True, exist_ok=False)
    audit_path = output / "audit.json"
    audit = {
        "collection_id": stamp,
        "status": "running",
        "dataset": DATASET,
        "geo": GEO,
        "geography_note": "EA20 fixed composition backcast, not changing-composition headline EA",
        "unit": "I21",
        "seasonal_adjustment": "NSA",
        "lag_months": LAG_MONTHS,
        "availability_rule": "HICP last observed month t uses PPI reference month t-2 and earlier",
        "release_dates_reconstructed": False,
        "historical_vintages_available": False,
        "interpretation": "Retrospective fixed-vintage features with conservative lag proxy; revisions remain possible",
        "imputation": False,
        "sources": [],
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(prepare_data.__file__),
                Path(eurostat_client.__file__),
            ]
        },
    }

    def save():
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save()
    frames = []
    try:
        for name, sector in SECTORS.items():
            print(f"Téléchargement : {name} ({GEO}, {sector})", flush=True)
            filters = {
                "freq": "M",
                "indic_bt": "PRC_PRR_DOM",
                "nace_r2": sector,
                "s_adj": "NSA",
                "unit": "I21",
                "geo": GEO,
                "sinceTimePeriod": config.START_PERIOD,
                "untilTimePeriod": config.END_PERIOD,
            }
            path = download_dataset(DATASET, filters)
            frame, metadata = decode_snapshot(path, filters)
            frame["indicator"] = name
            frames.append(frame)
            valid = frame.loc[frame["value"].notna(), "month"]
            audit["sources"].append(
                {
                    "indicator": name,
                    "filters": filters,
                    "raw_file": str(path.relative_to(config.PROJECT_ROOT)),
                    "sha256": metadata["sha256"],
                    "metadata_file": str(
                        (path.parent / "metadata.json").relative_to(config.PROJECT_ROOT)
                    ),
                    "dataset_updated": metadata["dataset_updated"],
                    "retrieved_at_utc": metadata["retrieved_at_utc"],
                    "first_observed": str(valid.min()),
                    "last_observed": str(valid.max()),
                    "observations": len(valid),
                }
            )
            save()
        raw_frame = pd.concat(frames, ignore_index=True)
        indices = raw_frame.pivot(
            index="month", columns="indicator", values="value"
        ).astype(float)
        origins = pd.period_range(config.START_PERIOD, config.END_PERIOD, freq="M")
        features = make_features(indices, origins)
        complete = features.notna().all(axis=1)
        if not complete.any():
            raise ValueError("Aucun mois ne possède les six variables externes.")
        # Les débuts manquants sont documentés ; les trous à partir de la première
        # ligne complète sont bloquants pour le futur modèle enrichi.
        first = complete.loc[complete].index[0]
        if not complete.loc[first:].all():
            raise ValueError("Des trous subsistent dans la période externe commune.")
        raw_frame = raw_frame.drop(columns="month")
        raw_frame.to_parquet(output / "indices.parquet", index=False)
        features.index = features.index.astype(str)
        features.index.name = "origin_month"
        features.to_parquet(output / "features.parquet")
        audit.update(
            status="passed",
            feature_columns=list(features.columns),
            feature_months=int(complete.sum()),
            first_complete_origin=str(first),
            last_complete_origin=str(origins[-1]),
            prepared_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [output / "indices.parquet", output / "features.parquet"]
            },
        )
        save()
        (output / "_SUCCESS").write_text("passed\n", encoding="utf-8")
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        print("Données préparées :", output)
    except (Exception, KeyboardInterrupt) as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    return output


if __name__ == "__main__":
    main()
