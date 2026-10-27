"""Collecte des indices et pondérations."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from eurostat_client import download_dataset


def build_jobs() -> list[dict]:
    countries = sorted(
        country
        for country, entry in config.EURO_AREA_ENTRY.items()
        if entry <= config.END_PERIOD
    )

    geographies = [config.EURO_AREA, *countries]
    jobs = []

    for geo in geographies:
        # Toutes les composantes disponibles pour ce territoire.
        jobs.append({
            "name": f"indices_{geo}",
            "dataset": config.DATASETS["indices_and_rates"],
            "filters": {
                "freq": config.FREQUENCY,
                "geo": geo,
                "unit": config.INDEX_UNIT,
                "sinceTimePeriod": config.START_PERIOD,
                "untilTimePeriod": config.END_PERIOD,
            },
            "status": "pending",
        })

        jobs.append({
            "name": f"item_weights_{geo}",
            "dataset": config.DATASETS["item_weights"],
            "filters": {
                "freq": "A",
                "geo": geo,
                "statinfo": config.ITEM_WEIGHT_STATINFO,
                "sinceTimePeriod": config.START_PERIOD[:4],
                "untilTimePeriod": config.END_PERIOD[:4],
            },
            "status": "pending",
        })

    jobs.append({
        "name": "country_weights",
        "dataset": config.DATASETS["country_weights"],
        "filters": {
            "freq": "A",
            "statinfo": config.COUNTRY_WEIGHT_STATINFO,
            "sinceTimePeriod": config.START_PERIOD[:4],
            "untilTimePeriod": config.END_PERIOD[:4],
        },
        "status": "pending",
    })

    return jobs


def write_manifest(path: Path, manifest: dict) -> None:
    """Met à jour le manifeste en remplacement le fichier temporaire."""
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(path)


def main() -> None:
    now = datetime.now(timezone.utc)
    collection_id = now.strftime("%Y%m%dT%H%M%S_%fZ")

    manifest_dir = config.RAW_DATA_DIR / "collections"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"panel_{collection_id}.json"

    manifest = {
        "collection_id": collection_id,
        "created_at_utc": now.isoformat(),
        "status": "running",
        "start_period": config.START_PERIOD,
        "end_period": config.END_PERIOD,
        "classification": config.CLASSIFICATION,
        "index_unit": config.INDEX_UNIT,
        "euro_area_entry": config.EURO_AREA_ENTRY,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "jobs": build_jobs(),
    }

    write_manifest(manifest_path, manifest)
    print(f"Manifeste : {manifest_path}", flush=True)

    total = len(manifest["jobs"])

    for number, job in enumerate(manifest["jobs"], start=1):
        print(f"[{number}/{total}] {job['name']}", flush=True)

        job["status"] = "running"
        write_manifest(manifest_path, manifest)

        try:
            raw_path = download_dataset(
                dataset=job["dataset"],
                filters=job["filters"],
            )

            metadata_path = raw_path.parent / "metadata.json"
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

            job.update({
                "status": "complete",
                "raw_file": str(
                    raw_path.relative_to(config.PROJECT_ROOT)
                ),
                "metadata_file": str(
                    metadata_path.relative_to(config.PROJECT_ROOT)
                ),
                "sha256": metadata["sha256"],
                "observation_count": metadata["observation_count"],
                "dataset_updated": metadata.get("dataset_updated"),
                "retrieved_at_utc": metadata["retrieved_at_utc"],
            })

        except (Exception, KeyboardInterrupt) as exc:
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "failed"

            write_manifest(manifest_path, manifest)
            print(
                f"Collecte interrompue. Manifeste : {manifest_path}"
            )
            raise

        write_manifest(manifest_path, manifest)

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    write_manifest(manifest_path, manifest)

    print(f"Collecte terminée : {total} extractions.")
    print(f"Manifeste complet : {manifest_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    main()
