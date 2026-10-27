"""Extractions"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

import config

LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def download_dataset(dataset: str, filters: dict[str, str]) -> Path:
    """Télécharge une extraction et conserve un instantané daté."""
    url = f"{config.EUROSTAT_API_URL.rstrip('/')}/{dataset}"
    params = {"format": "JSON", "lang": "EN", **filters}
    requested_at = datetime.now(timezone.utc)

    with httpx.Client(
        timeout=config.REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:
        for attempt in range(3):
            try:
                response = client.get(url, params=params)
                response.raise_for_status()

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in RETRYABLE_STATUS or attempt == 2:
                    raise

            except httpx.TransportError:
                if attempt == 2:
                    raise

            else:
                break

            delay = 2 ** attempt
            LOGGER.warning("Nouvelle tentative dans %s seconde(s).", delay)
            time.sleep(delay)

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("Eurostat n'a pas renvoyé un JSON valide.") from exc

    if not isinstance(payload, dict) or payload.get("class") != "dataset":
        raise ValueError(f"Réponse Eurostat inattendue : {str(payload)[:500]}")

    values = payload.get("value")

    if isinstance(values, dict):
        values = list(values.values())

    if not isinstance(values, list):
        raise ValueError("La réponse ne contient pas de valeurs exploitables.")

    observation_count = sum(value is not None for value in values)

    if observation_count == 0:
        raise ValueError("L'extraction ne contient aucune observation.")

    retrieved_at = datetime.now(timezone.utc)
    stamp = retrieved_at.strftime("%Y%m%dT%H%M%S_%fZ")

    snapshot_dir = config.RAW_DATA_DIR / dataset / stamp
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    raw_path = snapshot_dir / "data.json"
    raw_path.write_bytes(response.content)

    metadata = {
        "dataset": dataset,
        "request_url": str(response.url),
        "parameters": params,
        "requested_at_utc": requested_at.isoformat(),
        "retrieved_at_utc": retrieved_at.isoformat(),
        "dataset_updated": payload.get("updated"),
        "observation_count": observation_count,
        "sha256": hashlib.sha256(response.content).hexdigest(),
    }

    metadata_path = snapshot_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    LOGGER.info("%s observations enregistrées.", observation_count)

    return raw_path


def main() -> None:
    filters = {
        "freq": config.FREQUENCY,
        "geo": config.EURO_AREA,
        "coicop18": config.HEADLINE_COMPONENT,
        "unit": config.INDEX_UNIT,
        "sinceTimePeriod": config.START_PERIOD,
        "untilTimePeriod": config.END_PERIOD,
    }

    print("Téléchargement de l'indice global de la zone euro...", flush=True)

    raw_path = download_dataset(
        dataset=config.DATASETS["indices_and_rates"],
        filters=filters,
    )

    print(f"Données brutes : {raw_path}")
    print(f"Métadonnées : {raw_path.parent / 'metadata.json'}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    main()
