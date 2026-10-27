"""Conversion JSON-stat et contrôles série globale IPCH."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


def expand_vector(data, length: int) -> list:
    """Reconstitue les cellules + celles sans valeur."""
    if isinstance(data, dict):
        result = [None] * length

        for key, value in data.items():
            position = int(key)

            if not 0 <= position < length:
                raise ValueError("Position JSON-stat hors limites.")

            result[position] = value

        return result

    if isinstance(data, list):
        if len(data) != length:
            raise ValueError("Longueur JSON-stat incohérente.")

        return data

    if data is None or isinstance(data, str):
        return [data] * length

    raise ValueError("Représentation JSON-stat non prise en charge.")


def jsonstat_to_frame(payload: dict) -> pd.DataFrame:
    """Conserve l'ordre des dimensions, les libellés et les statuts."""
    if payload.get("class") != "dataset":
        raise ValueError("Le JSON ne contient pas un dataset JSON-stat.")

    dimensions = payload["id"]
    sizes = payload["size"]

    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("Dimensions JSON-stat invalides.")

    axes = []

    for name, size in zip(dimensions, sizes, strict=True):
        category = payload["dimension"][name]["category"]
        index = category["index"]

        if isinstance(index, dict):
            if sorted(index.values()) != list(range(size)):
                raise ValueError(f"Positions incohérentes pour {name}.")

            codes = sorted(index, key=index.get)

        elif isinstance(index, list):
            codes = index

        else:
            raise ValueError(f"Index non pris en charge pour {name}.")

        if size <= 0 or len(codes) != size or len(set(codes)) != size:
            raise ValueError(f"Catégories incohérentes pour {name}.")

        axes.append(codes)

    frame = pd.MultiIndex.from_product(
        axes, names=dimensions
    ).to_frame(index=False)

    frame["value"] = pd.array(
        expand_vector(payload["value"], len(frame)),
        dtype="Float64",
    )

    frame["status"] = pd.array(
        expand_vector(payload.get("status"), len(frame)),
        dtype="string",
    )

    for name in dimensions:
        labels = payload["dimension"][name]["category"].get("label", {})
        frame[f"{name}_label"] = frame[name].map(labels).astype("string")

    return frame


def load_snapshot(filters: dict) -> tuple[Path, dict, dict]:
    """Choisit le dernier instantané correspondant à la configuration."""
    directory = config.RAW_DATA_DIR / config.DATASETS["indices_and_rates"]

    for path in sorted(directory.glob("*/data.json"), reverse=True):
        metadata_path = path.parent / "metadata.json"

        if not metadata_path.is_file():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parameters = metadata.get("parameters", {})

        if not all(
            parameters.get(key) == value
            for key, value in filters.items()
        ):
            continue

        content = path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()

        if actual_hash != metadata["sha256"]:
            raise ValueError(
                f"L'empreinte du fichier ne correspond plus : {path}"
            )

        return path, metadata, json.loads(content)

    raise FileNotFoundError(
        "Aucun instantané compatible. Lance d'abord : "
        "uv run python eurostat_client.py"
    )


def main() -> None:
    filters = {
        "freq": config.FREQUENCY,
        "geo": config.EURO_AREA,
        "coicop18": config.HEADLINE_COMPONENT,
        "unit": config.INDEX_UNIT,
        "sinceTimePeriod": config.START_PERIOD,
        "untilTimePeriod": config.END_PERIOD,
    }

    source, metadata, payload = load_snapshot(filters)
    frame = jsonstat_to_frame(payload)

    # Vérifier la sélection effectivement reçue.
    for name in ("freq", "geo", "coicop18", "unit"):
        if not frame[name].eq(filters[name]).all():
            raise ValueError(
                f"La sélection reçue est incorrecte pour {name}."
            )

    frame["date"] = pd.to_datetime(
        frame["time"], format="%Y-%m"
    )
    frame = frame.sort_values("date").reset_index(drop=True)

    expected = pd.date_range(
        config.START_PERIOD,
        config.END_PERIOD,
        freq="MS",
    )
    dates = pd.DatetimeIndex(frame["date"])

    checks = {
        "duplicate_keys": int(
            frame.duplicated(subset=payload["id"]).sum()
        ),
        "missing_values": int(
            frame["value"].isna().sum()
        ),
        "non_positive_indices": int(
            frame["value"].le(0).sum()
        ),
        "infinite_values": int(
            np.isinf(
                frame["value"].to_numpy(
                    dtype=float, na_value=np.nan
                )
            ).sum()
        ),
        "missing_months": (
            expected.difference(dates)
            .strftime("%Y-%m")
            .tolist()
        ),
        "unexpected_months": (
            dates.difference(expected)
            .strftime("%Y-%m")
            .tolist()
        ),
    }

    passed = not any(checks.values())

    report = {
        "audit_status": "passed" if passed else "failed",
        "source_file": str(
            source.relative_to(config.PROJECT_ROOT)
        ),
        "source_metadata": metadata,
        "transform_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "pandas_version": pd.__version__,
        "rows": len(frame),
        "first_month": frame["time"].min(),
        "last_month": frame["time"].max(),
        "checks": checks,
        "status_counts": {
            str(code): int(count)
            for code, count in (
                frame["status"].dropna().value_counts().items()
            )
        },
    }

    output_dir = (
        config.PROCESSED_DATA_DIR
        / "headline"
        / source.parent.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Instantané utilisé : {source}")
    print(f"Audit : {report['audit_status']}")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"Rapport : {report_path}")

    if not passed:
        raise ValueError(
            "Contrôles échoués : consulter audit.json."
        )

    table_path = output_dir / "headline.parquet"
    frame.to_parquet(
        table_path,
        engine="pyarrow",
        index=False,
    )

    print(f"Tableau enregistré : {table_path}")

    columns = ["time", "geo", "coicop18", "value", "status"]
    print(frame[columns].tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
