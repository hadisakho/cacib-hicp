"""Validation rétrospective de prévisions HICP à 1, 2 et 3 mois.

Les 24 derniers mois de cibles restent réservés au test final.
L'origine désigne le dernier mois observé, pas une date historique de publication.
Les données sont celles du millésime collecté : ce n'est pas un backtest en vintages.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import config
import forecast_models
from baselines import future_months, level_to_yoy, validate_series
from forecast_models import MODEL_SPECS, forecast_levels
from prepare_panel import select_manifest

HORIZONS = (1, 2, 3)
VALIDATION_START = "2014-01"
FINAL_TEST_MONTHS = 24
MIN_TRAIN_MONTHS = 120


def validation_bounds(
    end_period: str,
    validation_start: str = VALIDATION_START,
    final_test_months: int = FINAL_TEST_MONTHS,
):
    if final_test_months < 1:
        raise ValueError("La période réservée doit contenir au moins un mois.")
    end = pd.Period(end_period, freq="M")
    test_start = end - final_test_months + 1
    start = pd.Period(validation_start, freq="M")
    if start >= test_start:
        raise ValueError("La validation doit précéder le test final.")
    return start, test_start - 1, test_start, end


def load_headline(manifest_path: Path | None = None):
    source_manifest, manifest = select_manifest(manifest_path)
    directory = config.PROCESSED_DATA_DIR / "panel" / manifest["collection_id"]
    audit_path = directory / "audit.json"
    if not (directory / "_SUCCESS").is_file() or not audit_path.is_file():
        raise FileNotFoundError(
            "Préparer d'abord cette collecte avec prepare_panel.py."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("integrity_status") != "passed":
        raise ValueError("Les contrôles bloquants du panel ne passent pas.")
    if config.HEADLINE_COMPONENT != "TOTAL":
        raise ValueError("Ce premier backtest porte sur l'indice global TOTAL.")
    path = directory / "indices" / f"{config.EURO_AREA}.parquet"
    frame = pd.read_parquet(path)
    selected = frame.loc[
        frame["geo"].eq(config.EURO_AREA)
        & frame["coicop18"].eq("TOTAL")
        & frame["unit"].eq(config.INDEX_UNIT)
        & frame["freq"].eq(config.FREQUENCY)
    ].copy()
    selected["month"] = pd.PeriodIndex(selected["time"], freq="M")
    selected = selected.sort_values("month")
    series = pd.Series(
        selected["value"].to_numpy(dtype=float, na_value=np.nan),
        index=pd.PeriodIndex(selected["month"], freq="M"),
        name="HICP_TOTAL",
    )
    series = validate_series(series)
    expected = pd.period_range(config.START_PERIOD, config.END_PERIOD, freq="M")
    if not series.index.equals(expected):
        raise ValueError(
            "La série globale ne couvre pas exactement la période configurée."
        )
    provenance = {
        "collection_id": manifest["collection_id"],
        "source_manifest": str(source_manifest.resolve()),
        "prepared_table": str(path.relative_to(config.PROJECT_ROOT)),
        "prepared_table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "panel_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    }
    return series, provenance


def rolling_backtest(
    history: pd.Series,
    models: list[str],
    target_start,
    target_end,
    min_train_months: int = MIN_TRAIN_MONTHS,
    window_months: int | None = None,
    forecaster=None,
    progress: bool = False,
) -> pd.DataFrame:
    """Chaque modèle reçoit uniquement une copie du préfixe observé à l'origine."""
    history = validate_series(history).copy()
    if not models or len(set(models)) != len(models):
        raise ValueError("Fournir une liste de modèles non vide, sans doublons.")
    if min_train_months < 13 or (
        window_months is not None and window_months < min_train_months
    ):
        raise ValueError("Fenêtre d'apprentissage insuffisante.")
    start, end = pd.Period(target_start, freq="M"), pd.Period(target_end, freq="M")
    if start > end or start < history.index[0] or end > history.index[-1]:
        raise ValueError("Période d'évaluation invalide.")
    targets = pd.period_range(start, end, freq="M")
    origins = sorted({target - horizon for target in targets for horizon in HORIZONS})
    if origins[0] not in history.index:
        raise ValueError("L'origine initiale est absente de l'historique.")
    if len(history.loc[: origins[0]]) < min_train_months:
        raise ValueError("Historique insuffisant à la première origine de validation.")
    predictor = forecast_levels if forecaster is None else forecaster
    records = []

    for number, origin in enumerate(origins, start=1):
        if progress and (number == 1 or number % 12 == 0 or number == len(origins)):
            print(f"Origine {origin} [{number}/{len(origins)}]", flush=True)
        train = history.loc[:origin].copy()
        if window_months is not None:
            train = train.iloc[-window_months:].copy()
        for model in models:
            prediction, error, details = None, None, {}
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    prediction = predictor(train.copy(), model, max(HORIZONS))
                    expected_index = future_months(train, max(HORIZONS))
                    if not isinstance(
                        prediction, pd.Series
                    ) or not prediction.index.equals(expected_index):
                        raise ValueError(
                            "La trajectoire prévue n'a pas les mois attendus."
                        )
                    values = prediction.to_numpy(dtype=float)
                    if not np.isfinite(values).all() or (values <= 0).any():
                        raise ValueError(
                            "La trajectoire prévue contient des indices invalides."
                        )
                    details = prediction.attrs.get("fit_details", {})
                except (
                    ValueError,
                    RuntimeError,
                    ArithmeticError,
                    np.linalg.LinAlgError,
                ) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            warning_messages = list(dict.fromkeys(str(w.message) for w in captured))

            for horizon in HORIZONS:
                target = origin + horizon
                if not start <= target <= end:
                    continue
                base = float(train.loc[target - 12])
                actual_index = float(history.loc[target])
                actual_yoy = 100.0 * (actual_index / base - 1.0)
                predicted_index = (
                    float(prediction.loc[target]) if error is None else np.nan
                )
                predicted_yoy = (
                    level_to_yoy(train, predicted_index, target)
                    if error is None
                    else np.nan
                )
                records.append(
                    {
                        "model": model,
                        "origin": str(origin),
                        "target": str(target),
                        "horizon": horizon,
                        "train_start": str(train.index[0]),
                        "train_end": str(train.index[-1]),
                        "train_months": len(train),
                        "base_month": str(target - 12),
                        "base_index": base,
                        "actual_index": actual_index,
                        "predicted_index": predicted_index,
                        "actual_yoy": actual_yoy,
                        "predicted_yoy": predicted_yoy,
                        "error_pp": predicted_yoy - actual_yoy,
                        "status": "ok" if error is None else "failed",
                        "error_message": error,
                        "warnings": json.dumps(warning_messages, ensure_ascii=False),
                        "fit_details": json.dumps(details, ensure_ascii=False),
                    }
                )

    return pd.DataFrame.from_records(records)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Un modèle incomplet n'obtient pas de score sur une sélection de dates faciles."""
    rows = []
    for (model, horizon), group in predictions.groupby(["model", "horizon"], sort=True):
        valid = group["status"].eq("ok") & np.isfinite(group["error_pp"])
        complete = bool(valid.all())
        errors = group.loc[valid, "error_pp"].to_numpy()
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "expected_forecasts": len(group),
                "successful_forecasts": int(valid.sum()),
                "coverage": float(valid.mean()),
                "eligible": complete,
                "mae_pp": float(np.mean(np.abs(errors))) if complete else np.nan,
                "rmse_pp": float(np.sqrt(np.mean(errors**2))) if complete else np.nan,
                "bias_pp": float(np.mean(errors)) if complete else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "rmse_pp"]).reset_index(drop=True)


def main(
    manifest_path: Path | None = None,
    models: list[str] | None = None,
    window_months: int | None = None,
) -> Path:
    models = list(MODEL_SPECS) if models is None else models
    if any(model not in MODEL_SPECS for model in models):
        raise ValueError("Modèle inconnu.")
    history, provenance = load_headline(manifest_path)
    start, end, test_start, test_end = validation_bounds(config.END_PERIOD)
    run_id = datetime.now(UTC).strftime("validation_%Y%m%dT%H%M%S_%fZ")
    output = config.PROJECT_ROOT / "reports" / "backtests" / run_id
    output.mkdir(parents=True, exist_ok=False)
    source_files = {
        "baselines.py": Path(baselines.__file__),
        "forecast_models.py": Path(forecast_models.__file__),
        "backtest.py": Path(__file__),
    }
    experiment = {
        "run_id": run_id,
        "status": "running",
        "phase": "validation",
        "evaluation_type": "retrospective_fixed_vintage",
        "origin_definition": "last_observed_month; historical release dates not reconstructed",
        "target": "Euro area headline HICP year-on-year inflation, percent",
        "validation_targets": [str(start), str(end)],
        "reserved_final_test_targets": [str(test_start), str(test_end)],
        "horizons": list(HORIZONS),
        "minimum_training_months": MIN_TRAIN_MONTHS,
        "window_months": window_months,
        "primary_metric": "rmse_pp at horizon 3",
        "model_specs": {name: MODEL_SPECS[name] for name in models},
        "data": provenance,
        "python": platform.python_version(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pyarrow", "statsmodels")
        },
        "code_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files.items()
        },
    }
    metadata_path = output / "experiment.json"
    metadata_path.write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Validation : cibles {start} à {end}")
    print(f"Test final réservé : cibles {test_start} à {test_end}")
    print("Évaluation rétrospective sur le millésime collecté.", flush=True)
    try:
        predictions = rolling_backtest(
            history, models, start, end, window_months=window_months, progress=True
        )
        expected = (
            len(pd.period_range(start, end, freq="M")) * len(models) * len(HORIZONS)
        )
        if len(predictions) != expected:
            raise ValueError("La matrice des prévisions n'est pas complète.")
        scores = summarize(predictions)
        predictions.to_parquet(
            output / "predictions.parquet", engine="pyarrow", index=False
        )
        scores.to_csv(output / "metrics.csv", index=False, float_format="%.8f")
        failed = predictions.loc[predictions["status"].ne("ok")].drop_duplicates(
            ["model", "origin"]
        )
        failed.to_parquet(
            output / "fit_failures.parquet", engine="pyarrow", index=False
        )
        experiment["status"] = (
            "completed" if failed.empty else "completed_with_fit_failures"
        )
        experiment["failed_fits"] = len(failed)
        experiment["completed_at_utc"] = datetime.now(UTC).isoformat()
        print(scores.to_string(index=False), flush=True)
        print(f"Résultats : {output}")
    except (Exception, KeyboardInterrupt) as exc:
        experiment["status"] = "failed"
        experiment["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata_path.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=None)
    parser.add_argument(
        "--window-months",
        type=int,
        default=None,
        help="Nombre de mois d'apprentissage ; par défaut, fenêtre croissante.",
    )
    args = parser.parse_args()
    main(args.manifest, args.models, args.window_months)
