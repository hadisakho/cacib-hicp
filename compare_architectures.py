"""Validation directe contre cinq composantes agrégées, sur historique commun.

uv run python compare_architectures.py
Poids de l'année de l'origine supposés connus (origine = dernier mois publié).
Au passage de janvier, les derniers poids connus sont reconduits : pas de poids futurs.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import baselines
import config
import forecast_models
from check_aggregation import BRANCHES
from forecast_models import MODEL_SPECS, forecast_levels
from prepare_panel import select_manifest

DEFAULT_MODELS = ["flat_yoy", "seasonal_mean", "ets_additive"]


def aggregate_paths(history, forecasts, weights):
    """Agrège uniquement une histoire passée et les trajectoires prévues."""
    origin = history.index[-1]
    expected = pd.period_range(origin + 1, periods=len(forecasts), freq="M")
    if not forecasts.index.equals(expected) or not 1 <= len(forecasts) <= 3:
        raise ValueError("Une trajectoire de 1 à 3 mois consécutifs est attendue.")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("Historique dupliqué ou non trié.")
    if weights.index.has_duplicates:
        raise ValueError("Années de poids dupliquées.")
    known = weights.loc[weights.index <= origin.year, BRANCHES].sort_index()
    if origin.year not in known.index:
        raise ValueError("Poids de l'année de l'origine indisponibles.")
    w = known.loc[origin.year].astype(float)
    if not np.isfinite(w).all() or (w < 0).any() or abs(w.sum() - 1000) > 0.2:
        raise ValueError("Poids de l'origine invalides.")
    values = forecasts.loc[:, BRANCHES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prévisions de composantes invalides.")
    paths = pd.concat([history[BRANCHES], forecasts[BRANCHES]])
    headline = history["TOTAL"].to_dict()
    result = []
    for month in expected:
        base = pd.Period(f"{month.year - 1}-12", freq="M")
        if base not in paths.index or base not in headline:
            raise ValueError("Base de décembre absente de l’historique.")
        bases = paths.loc[base, BRANCHES].astype(float)
        if not np.isfinite(bases).all() or (bases <= 0).any():
            raise ValueError("Indices de décembre indisponibles.")
        factor = float(((paths.loc[month, BRANCHES] / bases) * w).sum() / w.sum())
        headline[month] = float(headline[base]) * factor
        result.append(headline[month])
    prediction = pd.Series(result, index=expected, dtype=float)
    prediction.attrs["fit_details"] = {
        "weights_year": int(origin.year),
        "future_year_weights": "carry_forward_origin_year",
        "crosses_january": bool(expected[-1].year > origin.year),
    }
    return prediction


def make_predictor(panel, weights, model_predictor=forecast_levels):
    def predict(train, name, horizon):
        architecture, model = name.split("__", 1)
        if architecture == "direct":
            return model_predictor(train, model, horizon)
        if architecture != "bottom_up":
            raise ValueError("Architecture inconnue.")
        # Le modèle reçoit le même intervalle historique que son concurrent direct.
        past = panel.loc[train.index].copy()
        future = pd.DataFrame(index=baselines.future_months(train, horizon))
        details = {}
        for component in BRANCHES:
            predicted = model_predictor(past[component].copy(), model, horizon)
            future[component] = predicted
            details[component] = predicted.attrs.get("fit_details", {})
        result = aggregate_paths(
            past, future, weights.loc[weights.index <= train.index[-1].year]
        )
        result.attrs["fit_details"]["components"] = details
        return result

    return predict


def load_panel():
    source_manifest, manifest = select_manifest()
    directory = config.PROCESSED_DATA_DIR / "panel" / manifest["collection_id"]
    if not (directory / "_SUCCESS").is_file():
        raise RuntimeError("Exécuter prepare_panel.py.")
    audit = directory / "audit.json"
    if json.loads(audit.read_text())["integrity_status"] != "passed":
        raise ValueError("Audit du panel non validé.")
    index_path = directory / "indices" / f"{config.EURO_AREA}.parquet"
    weight_path = directory / "item_weights" / f"{config.EURO_AREA}.parquet"
    frame = pd.read_parquet(index_path)
    frame = frame.loc[
        frame["coicop18"].isin(["TOTAL", *BRANCHES])
        & frame["geo"].eq(config.EURO_AREA)
        & frame["unit"].eq(config.INDEX_UNIT)
        & frame["freq"].eq(config.FREQUENCY)
    ].copy()
    frame["month"] = pd.PeriodIndex(frame["time"], freq="M")
    panel = frame.pivot(index="month", columns="coicop18", values="value").astype(float)
    start = max(panel[c].first_valid_index() for c in ["TOTAL", *BRANCHES])
    panel = panel.loc[start:]
    expected = pd.period_range(start, config.END_PERIOD, freq="M")
    if not panel.index.equals(expected):
        raise ValueError("Couverture mensuelle commune incorrecte.")
    for component in ["TOTAL", *BRANCHES]:
        baselines.validate_series(panel[component])
    raw_weights = pd.read_parquet(weight_path)
    raw_weights = raw_weights.loc[raw_weights["coicop18"].isin(BRANCHES)]
    weights = raw_weights.pivot(
        index="year", columns="coicop18", values="value"
    ).astype(float)
    provenance = {
        "collection_id": manifest["collection_id"],
        "source_manifest": str(source_manifest),
        "common_history_start": str(start),
        "sha256": {
            str(p.relative_to(config.PROJECT_ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [index_path, weight_path, audit]
        },
    }
    return panel, weights, provenance


def main(models=None):
    models = DEFAULT_MODELS if models is None else models
    if (
        not models
        or len(set(models)) != len(models)
        or any(m not in MODEL_SPECS for m in models)
    ):
        raise ValueError("Liste de modèles invalide.")
    panel, weights, provenance = load_panel()
    start, end, test_start, test_end = backtest.validation_bounds(config.END_PERIOD)
    names = [
        f"{architecture}__{model}"
        for model in models
        for architecture in ["direct", "bottom_up"]
    ]
    run_id = datetime.now(UTC).strftime("architectures_%Y%m%dT%H%M%S_%fZ")
    destination = config.PROJECT_ROOT / "reports" / "backtests" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    experiment = {
        "status": "running",
        "run_id": run_id,
        "evaluation_type": "retrospective_fixed_vintage",
        "validation_targets": [str(start), str(end)],
        "reserved_test_targets": [str(test_start), str(test_end)],
        "models": {m: MODEL_SPECS[m] for m in models},
        "training": "expanding, common history for both architectures",
        "minimum_training_months": backtest.MIN_TRAIN_MONTHS,
        "origin": "last observed month; release dates not reconstructed",
        "weights_assumption": "origin-year weights known; carry forward across January",
        "primary_metric": "RMSE in percentage points at horizon 3",
        "data": provenance,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "pyarrow", "statsmodels"]
        },
        "code_sha256": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [
                __file__,
                backtest.__file__,
                baselines.__file__,
                forecast_models.__file__,
            ]
        },
    }
    metadata_path = destination / "experiment.json"

    def save_metadata():
        metadata_path.write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save_metadata()
    try:
        print("Validation :", start, "à", end, flush=True)
        predictions = backtest.rolling_backtest(
            panel["TOTAL"],
            names,
            start,
            end,
            forecaster=make_predictor(panel, weights),
            progress=True,
        )
        expected_count = (
            len(pd.period_range(start, end, freq="M"))
            * len(names)
            * len(backtest.HORIZONS)
        )
        if len(predictions) != expected_count:
            raise ValueError("Matrice de prévisions incomplète.")
        predictions["crosses_january"] = (
            predictions["origin"].str[:4] != predictions["target"].str[:4]
        )
        scores = backtest.summarize(predictions)
        predictions.to_parquet(destination / "predictions.parquet", index=False)
        scores.to_csv(destination / "metrics.csv", index=False)
        failures = predictions.loc[predictions["status"].ne("ok")]
        failures.to_parquet(destination / "fit_failures.parquet", index=False)
        # La comparaison est également séparée selon le franchissement de janvier.
        for crosses, group in predictions.groupby("crosses_january"):
            backtest.summarize(group).to_csv(
                destination / f"metrics_crosses_january_{str(crosses).lower()}.csv",
                index=False,
            )
        experiment.update(
            status="completed" if failures.empty else "completed_with_failures",
            failed_forecasts=len(failures),
        )
        print(scores.loc[scores["horizon"].eq(3)].to_string(index=False), flush=True)
        print("Rapports :", destination)
    except BaseException as exc:
        experiment.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_metadata()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_SPECS), default=DEFAULT_MODELS
    )
    main(parser.parse_args().models)
