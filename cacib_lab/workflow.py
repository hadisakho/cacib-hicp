"""Audit, sélection figée, test indépendant et export du dashboard."""

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import (DIVISIONS, DIVISION_LABELS, Protocol, digest, initialize,
                   lab_root, load_snapshot, month, numerical_fingerprint,
                   project_root, read_json, stable_hash, tail_history, utcnow, write_json)
from .evaluation import (MODELS, ForecastEngine, load_run, metrics, run_comparison)
from .hierarchy import historical_diagnostics, leaf_paths, reconcile
from .uncertainty import bands, paired_block_bootstrap, prequential_intervals


def frame_records(frame):
    """Conversion standard : types numpy et NaN deviennent scalaires et null."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


def audit(snapshot):
    output = lab_root() / "audit"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Contrôles de disponibilité permis sur toute l'extraction; les graphiques
    # économiques et les diagnostics d'agrégation restent limités à la validation.
    for geo, component in snapshot.levels.columns:
        x = snapshot.series(geo, component)
        valid = x.where(np.isfinite(x) & x.gt(0))
        observed = valid.dropna()
        first, last = (observed.index[0], observed.index[-1]) if len(observed) else (None, None)
        interior = int(valid.loc[first:last].isna().sum()) if first is not None else 0
        rows.append({"geo": geo, "component": component,
                     "positive_observations": len(observed), "first_positive": str(first) if first else None,
                     "last_positive": str(last) if last else None,
                     "internal_gaps": interior, "nonpositive": int(x.le(0).sum()),
                     "trailing_contiguous_months": len(tail_history(x, snapshot.protocol.as_of)),
                     "policy": "trailing_complete_segment_then_logged_causal_fallback"})
    quality = pd.DataFrame(rows)
    quality.to_csv(output / "series_quality.csv", index=False)
    diagnostics = historical_diagnostics(snapshot)
    diagnostics.to_csv(output / "historical_reconstruction.csv", index=False)
    summary = []
    for name, g in diagnostics.groupby("architecture"):
        e = g.loc[g.status.eq("ok"), "error_index"]
        summary.append({"architecture": name, "months": len(g), "available": len(e),
                        "unavailable": int(g.status.ne("ok").sum()),
                        "mae_index": float(e.abs().mean()) if len(e) else None,
                        "max_abs_index": float(e.abs().max()) if len(e) else None})
    write_json(output / "summary.json", {"quality_series": len(quality),
               "series_with_internal_gaps": int(quality.internal_gaps.gt(0).sum()),
               "nonpositive_cells": int(quality.nonpositive.sum()),
               "reconstruction": summary,
               "diagnostics_end": snapshot.protocol.validation_end,
               "future_weights": "origin_year_weights_and_membership_carried_forward",
               "source_observations_modified": False})
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"Audit : {output}", flush=True)
    return output


def analyze(snapshot, run=None, window=None, calibration=None):
    output, meta, frame = load_run(run, window=window)
    leaves = pd.read_parquet(output / "leaf_predictions.parquet")
    other_frame = other_leaves = None
    if calibration is not None:
        path, _, other_frame = load_run(calibration)
        other_leaves = pd.read_parquet(path / "leaf_predictions.parquet")
    details, scores = prequential_intervals(snapshot, frame, leaves, other_frame, other_leaves)
    details.to_csv(output / "interval_predictions.csv", index=False)
    scores.to_csv(output / "interval_metrics.csv", index=False)
    paired_block_bootstrap(frame, seed=snapshot.protocol.seed).to_csv(
        output / "paired_loss_bootstrap.csv", index=False)
    # Diagnostics descriptifs; à h>1 les erreurs se chevauchent naturellement.
    from statsmodels.stats.diagnostic import acorr_ljungbox
    residuals = []
    for (model, h), g in frame.groupby(["model", "horizon"]):
        if not g.status.eq("ok").all() or len(g) < 25:
            continue
        x = g.sort_values("target")["error_pp"]
        diagnostic = acorr_ljungbox(x, lags=[12], return_df=True)
        residuals.append({"model": model, "horizon": int(h), "bias_pp": float(x.mean()),
                          "lag1_autocorrelation": float(x.autocorr()),
                          "ljung_box_lag12_pvalue": float(diagnostic.lb_pvalue.iloc[0]),
                          "note": "descriptive; multi-step forecast errors overlap"})
    pd.DataFrame(residuals).to_csv(output / "residual_diagnostics.csv", index=False)
    write_json(output / "analysis.json", {"completed_at": utcnow(),
               "interval_rule": "past_matured_out_of_sample_errors_only",
               "interval_minimum_origins": snapshot.protocol.min_calibration,
               "coverage_is_empirical_not_guaranteed": True,
               "bootstrap": {"block_months": 6, "draws": 2000},
               "interval_evaluation_rows": len(details)})
    print(f"Analyse et incertitude : {output}", flush=True)
    return output


def freeze(snapshot, run=None, window=None, model=None, replace=False):
    directory, metadata, frame = load_run(run, window=window)
    p = snapshot.protocol
    if metadata["phase"] != "validation" or metadata["targets"] != [p.validation_start, p.validation_end]:
        raise ValueError("La sélection exige la validation complète, sans cible de test.")
    if metadata["numerics"] != numerical_fingerprint() or metadata["inputs"] != stable_hash(snapshot.identity):
        raise ValueError("Le code ou les données diffèrent de la comparaison. Relancer la validation.")
    score = metrics(frame).query("horizon == 3 and eligible").sort_values(["rmse_pp", "model"])
    if score.empty:
        raise ValueError("Aucun candidat n'a produit toute la grille attendue.")
    chosen = model or str(score.iloc[0].model)
    if chosen not in set(score.model):
        raise ValueError("Le candidat choisi est absent ou comporte des échecs.")
    selection = {"model": chosen, "window": metadata["window"],
                 "criterion": "minimum_validation_RMSE_h3" if model is None else "explicit_pre_test_choice",
                 "validation_run": str(directory.relative_to(lab_root())),
                 "validation_signature": metadata["signature"],
                 "inputs": stable_hash(snapshot.identity), "numerics": numerical_fingerprint(),
                 "protocol": asdict(p),
                 "final_models": list(dict.fromkeys([chosen, "direct__flat_yoy", "direct__ets_additive"])),
                 "refit_rule": "rolling_refit_using_only_observations_through_origin",
                 "calibration_rule": "joint_matured_errors_update_without_parameter_selection"}
    path = lab_root() / "selection.json"
    if path.exists() and read_json(path) != selection:
        if (lab_root() / "test_opened.json").exists():
            raise ValueError("Le test a déjà été ouvert : la sélection de cette expérience est figée.")
        if not replace:
            raise ValueError("Une sélection existe. Avant le test seulement, --replace permet un choix explicite différent.")
    write_json(path, selection)
    print(f"Sélection figée : {chosen}. Test final non exécuté par cette commande.", flush=True)
    return selection


def verify_selection(snapshot):
    s = read_json(lab_root() / "selection.json")
    if s["inputs"] != stable_hash(snapshot.identity) or s["numerics"] != numerical_fingerprint():
        raise ValueError("Les calculs ou les données diffèrent du gel de sélection.")
    directory, meta, _ = load_run(lab_root() / s["validation_run"])
    if meta["signature"] != s["validation_signature"]:
        raise ValueError("La validation figée a été remplacée.")
    return s


def final_test(snapshot):
    selection = verify_selection(snapshot)
    identity = stable_hash(selection)
    opening = lab_root() / "test_opened.json"
    if opening.exists() and read_json(opening)["selection_hash"] != identity:
        raise ValueError("Le test final est associé à une autre sélection.")
    if not opening.exists():
        write_json(opening, {"selection_hash": identity, "opened_at": utcnow()})
    destination = run_comparison(snapshot, phase="test", models=selection["final_models"],
                                 window=selection["window"])
    analyze(snapshot, destination, calibration=lab_root() / selection["validation_run"])
    write_json(lab_root() / "final_test.json", {"selection_hash": identity,
               "directory": str(destination.relative_to(lab_root())),
               "completed_at": utcnow(), "test_used_for_reselection": False})
    return destination


def build_forecast(snapshot):
    selection = verify_selection(snapshot)
    receipt = read_json(lab_root() / "final_test.json")
    if receipt["selection_hash"] != stable_hash(selection):
        raise ValueError("Le test final ne correspond pas à la sélection.")
    val_dir, _, validation = load_run(lab_root() / selection["validation_run"])
    test_dir, _, test = load_run(lab_root() / receipt["directory"])
    calibration = pd.concat([validation, test], ignore_index=True)
    leaf_calibration = pd.concat([pd.read_parquet(d / "leaf_predictions.parquet")
                                  for d in [val_dir, test_dir]], ignore_index=True)
    p = snapshot.protocol
    origin = month(p.as_of)
    engine = ForecastEngine(snapshot, origin, selection["window"])
    point = engine.forecast(selection["model"])
    leaf_point = engine.leaves.get(selection["model"], (None,))[0]
    intervals = bands(snapshot, origin, selection["model"], point, calibration,
                      leaf_calibration, leaf_point)
    if intervals.lower.isna().any():
        raise ValueError("Calibration insuffisante pour les intervalles de la prévision finale.")
    forecasts = []
    for h, target in enumerate(point.index, 1):
        record = {"target": str(target), "horizon": h,
                  "predicted_index": float(point.loc[target]),
                  "predicted_yoy": float(100*(point.loc[target]/snapshot.headline.loc[target-12]-1))}
        for q in [.8, .95]:
            row = intervals.loc[intervals.target.eq(str(target)) & intervals.level.eq(q)].iloc[0]
            record[f"lower_{int(100*q)}"] = float(row.lower)
            record[f"upper_{int(100*q)}"] = float(row.upper)
        forecasts.append(record)
    # Une trajectoire détaillée est nécessaire aux scénarios, même si le modèle
    # retenu prédit directement TOTAL. La réconciliation est une allocation
    # proportionnelle transparente, sans prétention causale.
    _, paths, history, weights, notes = leaf_paths(snapshot, origin, "countries", "seasonal_mean", selection["window"])
    weights = weights.loc[weights.gt(0)]
    paths = paths.reindex(columns=weights.index)
    history = history.reindex(columns=weights.index)
    reconciled, factors = reconcile(history, paths, weights, snapshot.headline.loc[:origin], point)
    output = lab_root() / "current"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forecasts).to_csv(output / "forecast.csv", index=False)
    intervals.drop(columns=["leaf_fallback"], errors="ignore").to_csv(output / "forecast_intervals.csv", index=False)
    hist = snapshot.headline.loc[:origin]
    historical = pd.DataFrame({"month": hist.index.astype(str), "index": hist.values,
                               "yoy": (100*(hist/hist.shift(12)-1)).values}).tail(84)
    bands_score = pd.read_csv(test_dir / "interval_metrics.csv")
    keys = list(weights.index)
    past_months = history.index[-13:]
    bundle = {
        "schema_version": 1,
        "metadata": {"as_of": p.as_of, "generated_at": utcnow(),
                     "selected_model": selection["model"],
                     "collection_id": snapshot.identity["collection_id"],
                     "protocol": asdict(p), "window_months": selection["window"],
                     "evaluation_type": "retrospective_fixed_vintage",
                     "reconciliation_factors": factors,
                     "scenario_baseline": "country_division_seasonal_paths_proportionally_reconciled_to_selected_headline",
                     "interval_method": intervals.method.iloc[0],
                     "calibration_origins": int(intervals.calibration_origins.iloc[0]),
                     "limitations": [
                         "Millésime téléchargé : révisions historiques non reconstituées.",
                         "Origine = dernier mois observé, calendrier de publication non reconstitué.",
                         "PPI EA20 décalés de deux mois; périmètre distinct de la cible EA évolutive.",
                         "Poids et composition maintenus à leur dernière valeur connue dans les prévisions.",
                         "Intervalles empiriques; la couverture peut se dégrader pendant les ruptures.",
                         "Chocs conditionnels sur l'IPCH : transmission comptable, sans effets macroéconomiques indirects."]},
        "history": frame_records(historical), "forecast": forecasts,
        "validation_metrics": frame_records(metrics(validation)),
        "test_metrics": frame_records(metrics(test)),
        "interval_metrics": frame_records(bands_score),
        "validation_errors": frame_records(validation),
        "test_errors": frame_records(test),
        "countries": {g: snapshot.labels.get(g, g) for g in sorted({g for g,c in keys})},
        "components": DIVISION_LABELS,
        "scenario": {"origin": str(origin), "columns": [list(k) for k in keys],
                     "weights": [float(weights[k]) for k in keys],
                     "future_months": reconciled.index.astype(str).tolist(),
                     "future_values": reconciled.to_numpy().tolist(),
                     "history_months": past_months.astype(str).tolist(),
                     "history_values": history.loc[past_months].to_numpy().tolist(),
                     "headline_months": hist.index[-13:].astype(str).tolist(),
                     "headline_values": hist.iloc[-13:].to_numpy().tolist(),
                     "fallback_notes": notes}}
    write_json(output / "dashboard.json", bundle)
    deploy = project_root() / "deploy" / "dashboard.json"
    write_json(deploy, bundle)
    write_json(output / "forecast_metadata.json", {"selection": selection,
               "dashboard_sha256": digest(output / "dashboard.json"), "generated_at": utcnow()})
    print(pd.DataFrame(forecasts).to_string(index=False), flush=True)
    print(f"Prévision : {output}. Export pour déploiement : {deploy}", flush=True)
    return output


def unpack_scenario(bundle):
    s = bundle["scenario"]
    columns = pd.MultiIndex.from_tuples([tuple(k) for k in s["columns"]], names=["geo", "component"])
    history = pd.DataFrame(s["history_values"], index=pd.PeriodIndex(s["history_months"], freq="M"), columns=columns)
    future = pd.DataFrame(s["future_values"], index=pd.PeriodIndex(s["future_months"], freq="M"), columns=columns)
    weights = pd.Series(s["weights"], index=columns)
    headline = pd.Series(s["headline_values"], index=pd.PeriodIndex(s["headline_months"], freq="M"))
    return history, future, weights, headline
