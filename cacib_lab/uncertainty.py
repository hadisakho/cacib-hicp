"""Calibration préquentielle et tirages conjoints de vecteurs d'erreurs.

Les quantiles sont empiriques : aucune garantie de couverture sous rupture
structurelle. Les erreurs dont la cible n'est pas encore observée sont exclues.
"""

import numpy as np
import pandas as pd
from .data import month
from .hierarchy import aggregate, linked_history, weights_at


def mature_errors(records, origin, model, cells=None):
    selected = records.loc[records["model"].eq(model)
                           & records["target"].le(str(month(origin)))].copy()
    valid = selected["actual_index"].gt(0) & selected["predicted_index"].gt(0)
    selected = selected.loc[valid]
    selected["log_error"] = np.log(selected["actual_index"] / selected["predicted_index"])
    if cells is None:
        matrix = selected.pivot(index="origin", columns="horizon", values="log_error")
        matrix = matrix.reindex(columns=[1, 2, 3])
    else:
        selected = selected.loc[pd.MultiIndex.from_frame(selected[["geo", "component"]]).isin(cells)]
        matrix = selected.pivot(index="origin", columns=["horizon", "geo", "component"], values="log_error")
        columns = pd.MultiIndex.from_tuples([(h, g, c) for h in [1, 2, 3] for g, c in cells])
        matrix = matrix.reindex(columns=columns)
    return matrix.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def volatility_factors(records, origin, model):
    """Élargissement causal si les douze dernières erreurs sont plus volatiles.

    Le facteur est au moins 1 : on n'utilise pas une accalmie récente pour
    rétrécir l'intervalle historique. Les mêmes facteurs s'appliquent à tous
    les tirages et toutes les cellules d'un horizon.
    """
    errors = mature_errors(records, origin, model)
    if len(errors) < 12:
        return np.ones(3)
    overall = np.sqrt(np.mean(errors.to_numpy()**2, axis=0))
    recent = np.sqrt(np.mean(errors.iloc[-12:].to_numpy()**2, axis=0))
    return np.maximum(1., recent / np.maximum(overall, 1e-10))


def joint_headline_draws(point, records, model, origin, min_calibration=36, draws=2000, seed=1729):
    errors = mature_errors(records, origin, model)
    if len(errors) < min_calibration:
        return None, {"method": "insufficient_calibration", "calibration_origins": len(errors)}
    rng = np.random.default_rng(seed + month(origin).ordinal)
    horizons = [t.ordinal - month(origin).ordinal - 1 for t in point.index]
    scales = volatility_factors(records, origin, model)
    sampled = (errors.to_numpy()[rng.integers(0, len(errors), size=draws)] * scales[None,:])[:, horizons]
    values = point.to_numpy()[None, :] * np.exp(sampled)
    return values, {"method": "joint_headline_log_errors_volatility_scaled", "calibration_origins": len(errors),
                    "volatility_scale": scales.tolist()}


def joint_leaf_draws(snapshot, origin, paths, model, leaf_records, min_calibration=36, draws=2000, seed=1729, scales=None):
    architecture = model.split("__", 1)[0]
    weights = weights_at(snapshot, month(origin).year, architecture)
    weights = weights.loc[weights.gt(0)]
    cells = list(weights.index)
    errors = mature_errors(leaf_records, origin, model, cells)
    if len(errors) < min_calibration:
        return None, {"method": "insufficient_joint_leaf_calibration", "calibration_origins": len(errors)}
    rng = np.random.default_rng(seed + month(origin).ordinal)
    sample = errors.to_numpy()[rng.integers(0, len(errors), size=draws)].reshape(draws, 3, len(cells))
    scales = np.ones(3) if scales is None else np.asarray(scales)
    sample *= scales[None,:,None]
    values = paths.reindex(columns=weights.index).to_numpy()[None, :, :] * np.exp(sample)
    history, _ = linked_history(snapshot, origin, weights.index)
    top_history = snapshot.headline.loc[:month(origin)]
    output = np.empty((draws, 3))
    for h, target in enumerate(paths.index):
        base = month(f"{target.year - 1}-12")
        if base > month(origin):
            k = paths.index.get_loc(base)
            bases = values[:, k, :]
            top_base = output[:, k]
        else:
            bases = history.loc[base].to_numpy()[None, :]
            top_base = float(top_history.loc[base])
        output[:, h] = top_base * np.sum(values[:, h, :] / bases * weights.to_numpy()[None, :], axis=1)
    return output, {"method": "joint_leaf_log_errors_then_chain_aggregation_volatility_scaled",
                    "calibration_origins": len(errors), "joint_cells": len(cells),
                    "volatility_scale": scales.tolist()}


def bands(snapshot, origin, model, point, records, leaf_records=None, leaf_paths=None):
    p = snapshot.protocol
    info = {}
    simulated = None
    if (model.startswith(("branches__", "countries__")) and leaf_records is not None
            and leaf_paths is not None and len(leaf_paths) == 3
            and leaf_paths.index[0] == month(origin) + 1):
        simulated, info = joint_leaf_draws(snapshot, origin, leaf_paths, model, leaf_records,
                                           p.min_calibration, p.interval_draws, p.seed,
                                           volatility_factors(records, origin, model))
    if simulated is None:
        previous = info
        simulated, info = joint_headline_draws(point, records, model, origin,
                                               p.min_calibration, p.interval_draws, p.seed)
        if previous:
            info["leaf_fallback"] = previous
    rows = []
    for h, target in enumerate(point.index):
        denominator = float(snapshot.headline.loc[target - 12])
        point_yoy = 100 * (point.loc[target] / denominator - 1)
        for level in [.8, .95]:
            if simulated is not None:
                alpha = 1 - level
                low, high = np.quantile(100 * (simulated[:, h] / denominator - 1),
                                        [alpha/2, 1-alpha/2])
                low, high = float(low), float(high)
            else:
                low = high = None
            rows.append({"origin": str(origin), "target": str(target), "model": model,
                         "horizon": target.ordinal-month(origin).ordinal, "level": level, "point_yoy": float(point_yoy),
                         "lower": low, "upper": high, **info})
    return pd.DataFrame(rows)


def interval_score(actual, lower, upper, level):
    alpha = 1-level
    return upper-lower + 2/alpha*max(lower-actual, 0) + 2/alpha*max(actual-upper, 0)


def prequential_intervals(snapshot, frame, leaves, calibration=None, calibration_leaves=None):
    pool = frame if calibration is None else pd.concat([calibration, frame], ignore_index=True)
    leaf_pool = leaves if calibration_leaves is None else pd.concat([calibration_leaves, leaves], ignore_index=True)
    rows = []
    for (model, origin), group in frame.groupby(["model", "origin"], sort=True):
        if not group["status"].eq("ok").all():
            continue
        group = group.sort_values("horizon")
        point = pd.Series(group["predicted_index"].to_numpy(),
                          index=pd.PeriodIndex(group["target"], freq="M"))
        paths = None
        if model.startswith(("branches__", "countries__")) and len(leaves):
            this = leaves.loc[leaves["model"].eq(model) & leaves["origin"].eq(origin)]
            paths = this.pivot(index="target", columns=["geo", "component"], values="predicted_index")
            paths.index = pd.PeriodIndex(paths.index, freq="M")
        result = bands(snapshot, origin, model, point, pool, leaf_pool, paths)
        result = result.loc[result["lower"].notna()].copy()
        if result.empty:
            continue
        actual = group.set_index("target")["actual_yoy"]
        result["actual_yoy"] = result["target"].map(actual)
        result["covered"] = result["actual_yoy"].between(result["lower"], result["upper"])
        result["width"] = result["upper"] - result["lower"]
        result["interval_score"] = [interval_score(a,l,u,q) for a,l,u,q in zip(
            result.actual_yoy, result.lower, result.upper, result.level)]
        # Les dictionnaires de diagnostic restent dans le JSON final, pas le CSV.
        rows.append(result.drop(columns=["leaf_fallback"], errors="ignore"))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    detail = pd.concat(rows, ignore_index=True)
    score = detail.groupby(["model", "horizon", "level"]).agg(
        evaluated=("covered", "size"), coverage=("covered", "mean"),
        mean_width=("width", "mean"), mean_interval_score=("interval_score", "mean")).reset_index()
    return detail, score


def paired_block_bootstrap(frame, benchmark="direct__ets_additive", block=6, draws=2000, seed=1729):
    eligible = frame.loc[frame["horizon"].eq(3)].groupby("model")["status"].apply(lambda s: s.eq("ok").all())
    eligible = eligible.loc[eligible].index
    pivot = frame.loc[frame["horizon"].eq(3) & frame["model"].isin(eligible)].pivot(
        index="target", columns="model", values="error_pp").sort_index()
    if benchmark not in pivot or pivot.isna().any().any():
        return pd.DataFrame()
    n = len(pivot)
    if n < block:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-block+1, size=(draws, int(np.ceil(n/block))))
    indices = (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :n]
    baseline = pivot[benchmark].to_numpy() ** 2
    rows = []
    for model in pivot:
        delta = pivot[model].to_numpy()**2 - baseline
        low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        rows.append({"model": model, "benchmark": benchmark, "targets": n,
                     "mean_squared_loss_difference": float(delta.mean()),
                     "ci95_low": float(low), "ci95_high": float(high),
                     "block_months": block, "interpretation": "negative_favors_candidate"})
    return pd.DataFrame(rows)
