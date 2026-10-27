"""Premiers modèles candidats ; spécifications fixées avant comparaison."""

import numpy as np
import pandas as pd

from baselines import baseline_forecast, future_months, validate_series

MODEL_SPECS = {
    "flat_index": {"description": "Indice constant"},
    "flat_yoy": {"description": "Glissement annuel constant"},
    "seasonal_mean": {
        "description": "Moyenne de trajectoires saisonnières",
        "years": 5,
    },
    "sarima_airline": {
        "order": [0, 1, 1],
        "seasonal_order": [0, 1, 1, 12],
        "transform": "log_index",
        "point": "conditional_gaussian_mean",
        "optimizers": ["lbfgs", "powell"],
    },
    "ets_additive": {
        "trend": "add",
        "damped_trend": True,
        "seasonal": "add",
        "seasonal_periods": 12,
        "transform": "index_level",
        "initialization_method": "estimated",
        "retry_with_brute_initialization": True,
    },
}


def forecast_levels(train: pd.Series, model: str, horizon: int = 3) -> pd.Series:
    train = validate_series(train)
    future = future_months(train, horizon)
    if model not in MODEL_SPECS:
        raise ValueError(f"Modèle inconnu : {model}")
    if model in ("flat_index", "flat_yoy", "seasonal_mean"):
        return baseline_forecast(
            train,
            model,
            horizon,
            seasonal_years=MODEL_SPECS["seasonal_mean"]["years"],
        )

    if len(train) < 36:
        raise ValueError("Ces modèles saisonniers nécessitent au moins 36 mois.")
    diagnostics = {}
    if model == "sarima_airline":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        spec = MODEL_SPECS[model]
        fitted_model = SARIMAX(
            np.log(train.to_numpy()),
            order=tuple(spec["order"]),
            seasonal_order=tuple(spec["seasonal_order"]),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        optimizer = spec["optimizers"][0]
        fitted = fitted_model.fit(disp=False, method=optimizer, maxiter=300)
        if not fitted.mle_retvals.get("converged", False):
            optimizer = spec["optimizers"][1]
            fitted = fitted_model.fit(
                start_params=fitted.params, disp=False, method=optimizer, maxiter=1000
            )
        if not fitted.mle_retvals.get("converged", False):
            raise RuntimeError("Le SARIMA n'a pas convergé.")
        distribution = fitted.get_forecast(steps=horizon)
        log_mean = np.asarray(distribution.predicted_mean)
        log_variance = np.asarray(distribution.var_pred_mean)
        if not np.isfinite(log_variance).all() or (log_variance < 0).any():
            raise ValueError("Variance de prévision SARIMA invalide.")
        # Moyenne d'une variable lognormale, conditionnellement aux paramètres.
        values = np.exp(log_mean + 0.5 * log_variance)
        diagnostics = {"optimizer": optimizer, "converged": True}
    elif model == "ets_additive":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        spec = MODEL_SPECS[model]
        fitted_model = ExponentialSmoothing(
            train.to_numpy(),
            trend=spec["trend"],
            damped_trend=spec["damped_trend"],
            seasonal=spec["seasonal"],
            seasonal_periods=spec["seasonal_periods"],
            initialization_method=spec["initialization_method"],
        )
        fitted = fitted_model.fit(optimized=True, use_brute=False)
        retry = (
            not bool(fitted.mle_retvals.get("success", False))
            and spec["retry_with_brute_initialization"]
        )
        if retry:
            fitted = fitted_model.fit(optimized=True, use_brute=True)
        if not bool(fitted.mle_retvals.get("success", False)):
            raise RuntimeError("Le modèle ETS n'a pas convergé.")
        values = np.asarray(fitted.forecast(horizon))
        diagnostics = {"converged": True, "brute_initialization_retry": retry}
    else:
        raise ValueError(f"Implémentation manquante pour {model}")

    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Le modèle a produit des indices invalides.")
    result = pd.Series(values, index=future, name="predicted_index", dtype=float)
    result.attrs["fit_details"] = diagnostics
    return result
