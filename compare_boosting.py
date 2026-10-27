"""Comparaison directe : boosting, Ridge, SARIMA, ETS et références simples.

uv run python compare_boosting.py
Les cinq composantes ne sont pas réestimées dans cette étape.
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
import boosting_model
import compare_architectures
import config
import forecast_models
import ridge_model
from boosting_model import BOOST_SPEC, boosting_forecast
from ridge_model import RIDGE_SPEC, ridge_forecast

MODELS = [
    "flat_yoy",
    "seasonal_mean",
    "sarima_airline",
    "ets_additive",
    "ridge_lags",
    "hist_boosting",
]


def predictor(train, name, horizon):
    if name == "ridge_lags":
        return ridge_forecast(train, horizon)
    if name == "hist_boosting":
        return boosting_forecast(train, horizon)
    return forecast_models.forecast_levels(train, name, horizon)


def main():
    panel, _, provenance = compare_architectures.load_panel()
    start, end, test_start, test_end = backtest.validation_bounds(config.END_PERIOD)
    run_id = datetime.now(UTC).strftime("direct_models_%Y%m%dT%H%M%S_%fZ")
    output = config.PROJECT_ROOT / "reports" / "backtests" / run_id
    output.mkdir(parents=True, exist_ok=False)
    specifications = {
        **forecast_models.MODEL_SPECS,
        "ridge_lags": RIDGE_SPEC,
        "hist_boosting": BOOST_SPEC,
    }
    files = [
        Path(__file__),
        *[
            Path(m.__file__)
            for m in (
                backtest,
                baselines,
                compare_architectures,
                forecast_models,
                ridge_model,
                boosting_model,
            )
        ],
    ]
    experiment = {
        "run_id": run_id,
        "status": "running",
        "phase": "validation",
        "architecture": "direct headline only",
        "evaluation_type": "retrospective_fixed_vintage",
        "validation_targets": [str(start), str(end)],
        "reserved_test_targets": [str(test_start), str(test_end)],
        "models": {m: specifications[m] for m in MODELS},
        "horizons": [1, 2, 3],
        "training": "same expanding common history as the architecture comparison",
        "minimum_training_months": backtest.MIN_TRAIN_MONTHS,
        "primary_metric": "RMSE in percentage points at horizon 3",
        "origin": "last observed month; publication dates not reconstructed",
        "data": provenance,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "pandas",
                "pyarrow",
                "statsmodels",
                "scikit-learn",
                "threadpoolctl",
            )
        },
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    metadata_path = output / "experiment.json"

    def save():
        metadata_path.write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save()
    try:
        print(
            f"Validation directe : {start} à {end} ; test final réservé : {test_start} à {test_end}.",
            flush=True,
        )
        predictions = backtest.rolling_backtest(
            panel["TOTAL"], MODELS, start, end, forecaster=predictor, progress=True
        )
        expected = len(pd.period_range(start, end, freq="M")) * len(MODELS) * 3
        if len(predictions) != expected:
            raise ValueError("Matrice de prévisions incomplète.")
        failures = predictions.loc[predictions["status"].ne("ok")]
        scores = backtest.summarize(predictions)
        predictions.to_parquet(output / "predictions.parquet", index=False)
        failures.to_parquet(output / "fit_failures.parquet", index=False)
        scores.to_csv(output / "metrics.csv", index=False)
        years = predictions["target"].str[:4].astype(int)
        predictions["period"] = pd.cut(
            years,
            [2013, 2019, 2020, 2023, 2024],
            labels=["2014-2019", "2020", "2021-2023", "2024-validation"],
        )
        period_scores = pd.concat(
            [
                backtest.summarize(g).assign(period=str(p))
                for p, g in predictions.groupby("period", observed=True)
            ],
            ignore_index=True,
        )
        period_scores.to_csv(output / "metrics_by_period.csv", index=False)
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
        save()
    return output


if __name__ == "__main__":
    main()
