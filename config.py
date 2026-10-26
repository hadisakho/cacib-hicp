"Configuration"

from pathlib import Path


# Racine
PROJECT_ROOT = Path(__file__).resolve().parent


# Horizon
FORECAST_HORIZON = 3  # Nombre de mois à prévoir


# Période d'extraction
START_PERIOD = "1996-01"
END_PERIOD = "2026-09"


# Sélection des données
FREQUENCY = "M"
EURO_AREA = "EA"  # Zone euro
CLASSIFICATION = "ECOICOP2"
HEADLINE_COMPONENT = "TOTAL"

INDEX_UNIT = "I25"       # Indice : moyenne de 2025 = 100
ANNUAL_RATE_UNIT = "RCH_A"  # Variation annuelle en %


# API Eurostat
EUROSTAT_API_URL = (
    "https://ec.europa.eu/eurostat/api/"
    "dissemination/statistics/1.0/data"
)

DATASETS = {
    "indices_and_rates": "prc_hicp_minr",
    "item_weights": "prc_hicp_iw",
    "country_weights": "prc_hicp_cw",
}

REQUEST_TIMEOUT_SECONDS = 60.0


# Emplacements des données
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


if __name__ == "__main__":
    print(f"Projet : {PROJECT_ROOT.name}")
    print(f"Période demandée : {START_PERIOD} à {END_PERIOD}")
    print(f"Horizon de prévision : {FORECAST_HORIZON} mois")
    print(f"Zone géographique : {EURO_AREA}")

    for name, dataset in DATASETS.items():
        print(f"{name} : {dataset}")
