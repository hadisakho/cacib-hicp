"""Gradient boosting avec les mêmes variables et cibles directes que Ridge."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from baselines import future_months, validate_series
from ridge_model import RIDGE_SPEC, inner_splits, make_design

BOOST_SPEC = {
    "estimator": "HistGradientBoostingRegressor",
    "target": "year-on-year inflation, separate direct model for each horizon",
    "features": RIDGE_SPEC["features"],
    "fixed_parameters": {
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": 100,
        "min_samples_leaf": 12,
        "max_bins": 32,
        "l2_regularization": 10.0,
        "early_stopping": False,
        "random_state": 42,
    },
    "candidate_max_leaf_nodes": [3, 7],
    "inner_splits": RIDGE_SPEC["inner_splits"],
    "inner_test_months": RIDGE_SPEC["inner_test_months"],
    "inner_gap": RIDGE_SPEC["inner_gap"],
    "inner_min_train_rows": RIDGE_SPEC["inner_min_train_rows"],
    "selection_metric": "mean squared error of year-on-year inflation",
    "threads": 1,
}


def boosting_forecast(train, horizon=3):
    train = validate_series(train)
    future = future_months(train, horizon)
    values, details = [], {}
    # Évite une surallocation de threads sur les petits historiques mensuels.
    with threadpool_limits(limits=1):
        for h in range(1, horizon + 1):
            x, y, query = make_design(train, h)
            splits = inner_splits(x, h)
            scores = {}
            for leaves in BOOST_SPEC["candidate_max_leaf_nodes"]:
                squared_errors = []
                for a, b in splits:
                    model = HistGradientBoostingRegressor(
                        max_leaf_nodes=leaves, **BOOST_SPEC["fixed_parameters"]
                    )
                    model.fit(x.iloc[a], y.iloc[a])
                    error = model.predict(x.iloc[b]) - y.iloc[b].to_numpy()
                    squared_errors.extend(error**2)
                scores[leaves] = float(np.mean(squared_errors))
            if not np.isfinite(list(scores.values())).all():
                raise ValueError("Scores de validation boosting non finis.")
            best = min(scores, key=scores.get)
            model = HistGradientBoostingRegressor(
                max_leaf_nodes=best, **BOOST_SPEC["fixed_parameters"]
            )
            model.fit(x, y)
            forecast_yoy = float(model.predict(query)[0])
            target = train.index[-1] + h
            level = float(train.loc[target - 12]) * (1 + forecast_yoy / 100)
            if not np.isfinite(level) or level <= 0:
                raise ValueError("Prévision boosting non positive ou non finie.")
            values.append(level)
            details[str(h)] = {
                "max_leaf_nodes": best,
                "inner_mse_by_max_leaf_nodes": {str(k): v for k, v in scores.items()},
                "train_rows": len(x),
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
    result = pd.Series(values, index=future, name="predicted_index", dtype=float)
    result.attrs["fit_details"] = {"boosting_by_horizon": details}
    return result
