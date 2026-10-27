"""Ridge avec PPI et solutions de repli explicites, sans PCA."""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import validate_series
from forecast_models import forecast_levels
from ridge_model import make_design, inner_splits, ridge_forecast, RIDGE_SPEC
from .data import month, tail_history


def ridge_external(train, external, horizon=3):
    """Les variables externes sont déjà indexées par origine, retard PPI inclus.

    Chaque ligne s utilise donc PPI(s-2). Les cibles d'apprentissage sont
    disponibles à l'origine courante; les plis internes purgent les labels h.
    """
    train = validate_series(train)
    if external is None:
        raise ValueError("Préparer les PPI pour utiliser ridge_ppi.")
    ext = external.loc[:train.index[-1]].copy()
    predictions, details = [], {}
    for h in range(1, horizon + 1):
        x, y, query = make_design(train, h)
        x = x.join(ext.add_prefix("external__"), how="left")
        query = query.join(ext.add_prefix("external__"), how="left")
        if not np.isfinite(query.to_numpy()).all():
            raise ValueError("Variables externes absentes à cette origine.")
        usable = np.isfinite(x.to_numpy()).all(axis=1)
        x, y = x.loc[usable], y.loc[usable]
        # Des lignes irrégulières modifieraient la signification du gap en mois.
        if not x.index.equals(pd.period_range(x.index.min(), x.index.max(), freq="M")):
            raise ValueError("Un trou interne persiste dans les variables externes.")
        splits = inner_splits(x, h)
        scores = {}
        for alpha in RIDGE_SPEC["alphas"]:
            errors = []
            for a, b in splits:
                pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="cholesky"))
                pipe.fit(x.iloc[a], y.iloc[a])
                errors.extend((pipe.predict(x.iloc[b]) - y.iloc[b].to_numpy()) ** 2)
            scores[alpha] = float(np.mean(errors))
        alpha = min(scores, key=scores.get)
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="cholesky"))
        pipe.fit(x, y)
        yoy = float(pipe.predict(query)[0])
        value = float(train.loc[train.index[-1] + h - 12]) * (1 + yoy / 100)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Prévision Ridge-PPI invalide.")
        predictions.append(value)
        details[str(h)] = {"alpha": alpha, "rows": len(x),
                           "last_label": str(x.index[-1] + h),
                           "folds": [{"last_train_label": str(x.index[a[-1]] + h),
                                      "first_validation_origin": str(x.index[b[0]])}
                                     for a, b in splits]}
    return pd.Series(predictions, index=pd.period_range(train.index[-1] + 1,
                     periods=horizon, freq="M")), details


def predict(series, origin, model, external=None, window=None, allow_fallback=False):
    """Le modèle voit uniquement le passé. Un repli est enregistré, jamais caché."""
    origin = month(origin)
    train = tail_history(series, origin, window)
    future = pd.period_range(origin + 1, periods=3, freq="M")
    try:
        if train.empty or train.index[-1] != origin:
            raise ValueError("Dernière observation indisponible.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model == "ridge":
                result = ridge_forecast(train, 3)
                info = result.attrs.get("fit_details", {})
            elif model == "ridge_ppi":
                result, info = ridge_external(train, external, 3)
            else:
                result = forecast_levels(train, model, 3)
                info = result.attrs.get("fit_details", {})
        info = {"requested": model, "used": model, "fallback": False,
                "warnings": sorted({str(w.message)[:250] for w in caught}),
                "details": info}
    except Exception as exc:
        if not allow_fallback:
            raise
        raw = series.loc[:origin]
        observed = raw.loc[np.isfinite(raw) & raw.gt(0)]
        if observed.empty:
            raise ValueError("Série jamais observée : aucun repli de niveau possible.") from exc
        # Une lacune récente ne reçoit pas une valeur issue du futur.
        if len(train) >= 13 and train.index[-1] == origin:
            result = forecast_levels(train, "flat_yoy", 3)
            used = "flat_yoy"
        else:
            result = pd.Series(float(observed.iloc[-1]), index=future)
            used = "last_positive_index"
        info = {"requested": model, "used": used, "fallback": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "last_observation": str(observed.index[-1])}
    if not result.index.equals(future) or not np.isfinite(result).all() or result.le(0).any():
        raise ValueError("Trajectoire prévue incorrecte.")
    return result.astype(float), info
