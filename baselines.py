"""Références de prévision des niveaux IPCH, aux horizons de 1 à 3 mois."""

import numpy as np
import pandas as pd


def validate_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.PeriodIndex) or series.index.freqstr != "M":
        raise ValueError("La série doit utiliser un PeriodIndex mensuel.")
    if (
        series.empty
        or not series.index.is_unique
        or not series.index.is_monotonic_increasing
    ):
        raise ValueError("La série doit être non vide, triée et sans doublons.")
    expected = pd.period_range(series.index[0], series.index[-1], freq="M")
    if not series.index.equals(expected):
        raise ValueError("La série comporte des mois absents.")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "La série modélisée doit être complète, finie et strictement positive."
        )
    return pd.Series(values, index=series.index, name=series.name)


def future_months(train: pd.Series, horizon: int) -> pd.PeriodIndex:
    if not 1 <= horizon <= 3:
        raise ValueError("L'horizon doit être compris entre 1 et 3 mois.")
    return pd.period_range(train.index[-1] + 1, periods=horizon, freq="M")


def baseline_forecast(
    train: pd.Series, model: str, horizon: int = 3, seasonal_years: int = 5
) -> pd.Series:
    train = validate_series(train)
    future = future_months(train, horizon)
    origin = train.index[-1]
    last = float(train.iloc[-1])

    if model == "flat_index":
        values = np.repeat(last, horizon)
    elif model == "flat_yoy":
        if origin - 12 not in train.index:
            raise ValueError("La persistance annuelle nécessite au moins 13 mois.")
        annual_factor = last / float(train.loc[origin - 12])
        values = [float(train.loc[target - 12]) * annual_factor for target in future]
    elif model == "seasonal_mean":
        if seasonal_years < 1 or origin - 12 * seasonal_years not in train.index:
            raise ValueError("Historique insuffisant pour la référence saisonnière.")
        # Moyenne arithmétique de trajectoires historiques relatives au même mois.
        # Chaque trajectoire couvre h mois et est déjà entièrement observée.
        values = [
            last
            * np.mean(
                [
                    float(train.loc[target - 12 * year])
                    / float(train.loc[origin - 12 * year])
                    for year in range(1, seasonal_years + 1)
                ]
            )
            for target in future
        ]
    else:
        raise ValueError(f"Référence inconnue : {model}")
    return pd.Series(values, index=future, name="predicted_index", dtype=float)


def level_to_yoy(train: pd.Series, predicted_index: float, target: pd.Period) -> float:
    """Le dénominateur annuel doit être déjà observé à l'origine."""
    base_month = target - 12
    if base_month > train.index[-1] or base_month not in train.index:
        raise ValueError("Le dénominateur annuel n'est pas disponible à l'origine.")
    if not np.isfinite(predicted_index) or predicted_index <= 0:
        raise ValueError("Prévision d'indice invalide.")
    return 100.0 * (float(predicted_index) / float(train.loc[base_month]) - 1.0)
