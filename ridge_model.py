"""Ridge par horizon, avec choix d'alpha dans le seul historique disponible."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import future_months, validate_series

RIDGE_SPEC = {
    "target": "year-on-year inflation, direct model for each horizon",
    "features": "12 monthly changes, 3 annual changes, known annual base effect, target-month sine/cosine",
    "alphas": [0.1, 1.0, 10.0, 100.0, 1000.0],
    "inner_splits": 3,
    "inner_test_months": 12,
    "inner_gap": "horizon - 1; last training label <= first validation origin",
    "inner_min_train_rows": 48,
    "selection_metric": "mean squared error of year-on-year inflation",
    "scaling": "StandardScaler fitted separately in each inner training fold",
    "solver": "cholesky",
}


def make_design(series, horizon):
    series = validate_series(series)
    future_months(series, horizon)  # Vérifie l'horizon.
    mom = 100 * (series / series.shift(1) - 1)
    yoy = 100 * (series / series.shift(12) - 1)
    features = pd.DataFrame(index=series.index)
    for lag in range(12):
        features[f"mom_lag_{lag}"] = mom.shift(lag)
    for lag in range(3):
        features[f"yoy_lag_{lag}"] = yoy.shift(lag)
    # Dénominateur annuel de la cible déjà connu à chaque origine s.
    features["known_annual_part"] = 100 * (series / series.shift(12 - horizon) - 1)
    target_month = (features.index + horizon).month
    features["calendar_sin"] = np.sin(2 * np.pi * target_month / 12)
    features["calendar_cos"] = np.cos(2 * np.pi * target_month / 12)
    labels = yoy.shift(-horizon)
    usable = features.notna().all(axis=1) & labels.notna()
    x, y = features.loc[usable], labels.loc[usable]
    query = features.iloc[[-1]]
    if query.isna().any().any():
        raise ValueError("Historique insuffisant pour construire les retards Ridge.")
    return x, y, query


def inner_splits(x, horizon):
    splitter = TimeSeriesSplit(
        n_splits=RIDGE_SPEC["inner_splits"],
        test_size=RIDGE_SPEC["inner_test_months"],
        gap=horizon - 1,
    )
    splits = list(splitter.split(x))
    for train_idx, validation_idx in splits:
        if len(train_idx) < RIDGE_SPEC["inner_min_train_rows"]:
            raise ValueError("Historique insuffisant pour la validation interne Ridge.")
        if x.index[train_idx[-1]] + horizon > x.index[validation_idx[0]]:
            raise ValueError(
                "Une cible d'apprentissage dépasse l'origine de validation."
            )
    return splits


def ridge_forecast(train, horizon=3):
    train = validate_series(train)
    future = future_months(train, horizon)
    forecasts, details = [], {}
    for h in range(1, horizon + 1):
        x, y, query = make_design(train, h)
        splits = inner_splits(x, h)
        scores = {}
        for alpha in RIDGE_SPEC["alphas"]:
            squared_errors = []
            for train_idx, validation_idx in splits:
                pipeline = make_pipeline(
                    StandardScaler(), Ridge(alpha=alpha, solver="cholesky")
                )
                pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])
                errors = (
                    pipeline.predict(x.iloc[validation_idx])
                    - y.iloc[validation_idx].to_numpy()
                )
                squared_errors.extend(errors**2)
            scores[alpha] = float(np.mean(squared_errors))
        if not np.isfinite(list(scores.values())).all():
            raise ValueError("Scores de validation interne non finis.")
        # À égalité, la première valeur de la grille est retenue de façon déterministe.
        best = min(scores, key=scores.get)
        fitted = make_pipeline(StandardScaler(), Ridge(alpha=best, solver="cholesky"))
        fitted.fit(x, y)
        predicted_yoy = float(fitted.predict(query)[0])
        target = train.index[-1] + h
        value = float(train.loc[target - 12]) * (1 + predicted_yoy / 100)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Ridge a produit un indice non positif ou non fini.")
        forecasts.append(value)
        details[str(h)] = {
            "alpha": best,
            "inner_mse_by_alpha": {str(a): score for a, score in scores.items()},
            "train_rows": len(x),
            "last_training_origin": str(x.index[-1]),
            "last_training_target": str(x.index[-1] + h),
            "folds": [
                {
                    "train_last_label": str(x.index[a[-1]] + h),
                    "validation_first_origin": str(x.index[b[0]]),
                    "validation_last_label": str(x.index[b[-1]] + h),
                }
                for a, b in splits
            ],
        }
    result = pd.Series(forecasts, index=future, name="predicted_index", dtype=float)
    result.attrs["fit_details"] = {"ridge_by_horizon": details}
    return result
