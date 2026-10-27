"""Compare Ridge et trois références, directement et par cinq composantes.

Utilisation : uv run python compare_ridge.py
Les modules précédents sont réutilisés sans modification.
"""

import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import backtest
import baselines
import check_aggregation
import compare_architectures
import config
import forecast_models
import ridge_model
from ridge_model import RIDGE_SPEC, ridge_forecast

MODELS = ["flat_yoy", "seasonal_mean", "ets_additive", "ridge_lags"]


def predictor(train, model, horizon):
    if model == "ridge_lags":
        return ridge_forecast(train, horizon)
    return forecast_models.forecast_levels(train, model, horizon)


def main():
    panel, weights, provenance = compare_architectures.load_panel()
    start, end, test_start, test_end = backtest.validation_bounds(config.END_PERIOD)
    names = [
        f"{architecture}__{model}"
        for model in MODELS
        for architecture in ("direct", "bottom_up")
    ]
    run_id = datetime.now(UTC).strftime("architectures_%Y%m%dT%H%M%S_%fZ")
    output = config.PROJECT_ROOT / "reports" / "backtests" / run_id
    output.mkdir(parents=True, exist_ok=False)
    code_files = [
        Path(__file__),
        *[
            Path(module.__file__)
            for module in (
                backtest,
                baselines,
                compare_architectures,
                forecast_models,
                ridge_model,
                check_aggregation,
            )
        ],
    ]
    experiment = {
        "run_id": run_id,
        "status": "running",
        "phase": "validation",
        "evaluation_type": "retrospective_fixed_vintage",
        "validation_targets": [str(start), str(end)],
        "reserved_test_targets": [str(test_start), str(test_end)],
        "models": {
            m: RIDGE_SPEC if m == "ridge_lags" else forecast_models.MODEL_SPECS[m]
            for m in MODELS
        },
        "horizons": [1, 2, 3],
        "primary_metric": "RMSE at horizon 3, percentage points",
        "training": "expanding, same common history for all models and architectures",
        "minimum_training_months": backtest.MIN_TRAIN_MONTHS,
        "origin": "last observed month; release dates not reconstructed",
        "weights_assumption": "origin-year weights known; carry forward across January",
        "ridge_tuning": "nested chronological folds, horizon-specific gap, training-only standardization",
        "data": provenance,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "pyarrow", "statsmodels", "scikit-learn")
        },
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files
        },
    }
    metadata_path = output / "experiment.json"

    def save_metadata():
        metadata_path.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save_metadata()
    try:
        print(
            f"Validation : {start} à {end}. Test final réservé : {test_start} à {test_end}.",
            flush=True,
        )
        predictions = backtest.rolling_backtest(
            panel["TOTAL"],
            names,
            start,
            end,
            forecaster=compare_architectures.make_predictor(
                panel, weights, model_predictor=predictor
            ),
            progress=True,
        )
        expected = len(pd.period_range(start, end, freq="M")) * len(names) * 3
        if len(predictions) != expected:
            raise ValueError("Matrice de prévisions incomplète.")
        predictions["crosses_january"] = (
            predictions["origin"].str[:4].ne(predictions["target"].str[:4])
        )
        scores = backtest.summarize(predictions)
        failures = predictions.loc[predictions["status"].ne("ok")]
        predictions.to_parquet(output / "predictions.parquet", index=False)
        failures.to_parquet(output / "fit_failures.parquet", index=False)
        scores.to_csv(output / "metrics.csv", index=False)
        for crosses, group in predictions.groupby("crosses_january"):
            backtest.summarize(group).to_csv(
                output / f"metrics_crosses_january_{str(crosses).lower()}.csv",
                index=False,
            )
        experiment.update(
            status="completed" if failures.empty else "completed_with_failures",
            failed_forecasts=len(failures),
            completed_at_utc=datetime.now(UTC).isoformat(),
        )
        print(scores.loc[scores["horizon"].eq(3)].to_string(index=False))
        print("Rapports :", output)
    except (Exception, KeyboardInterrupt) as exc:
        experiment.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_metadata()
    return output


if __name__ == "__main__":
    main()
