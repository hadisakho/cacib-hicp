"""Moteur unique de comparaison, reprise par origine et grille exhaustive."""

from datetime import UTC, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (BRANCHES, digest, lab_root, month, numerical_fingerprint,
                   read_json, stable_hash, tail_history, utcnow, write_json)
from .hierarchy import leaf_paths
from .models import predict

MODELS = ["direct__flat_yoy", "direct__seasonal_mean", "direct__ets_additive",
          "direct__ridge", "direct__ridge_ppi", "branches__ets_additive",
          "countries__seasonal_mean", "ensemble__ets_branches"]
MODEL_LABELS = {
    "direct__flat_yoy": "Inflation annuelle constante",
    "direct__seasonal_mean": "Profil saisonnier moyen",
    "direct__ets_additive": "ETS direct",
    "direct__ridge": "Ridge IPCH",
    "direct__ridge_ppi": "Ridge IPCH + PPI",
    "branches__ets_additive": "ETS - cinq composantes",
    "countries__seasonal_mean": "Saisonnier - pays × 13 divisions",
    "ensemble__ets_branches": "Moyenne ETS direct / composantes",
}


class ForecastEngine:
    def __init__(self, snapshot, origin, window=None):
        self.snapshot = snapshot
        self.origin = month(origin)
        self.window = window
        starts = [snapshot.series("EA", c).where(lambda s: s.gt(0)).first_valid_index()
                  for c in ["TOTAL", *BRANCHES]]
        if any(s is None for s in starts):
            raise ValueError("Une des cinq branches n'a aucun historique.")
        self.common_start = max(starts)
        self.cache = {}
        self.leaves = {}
        self.notes = {}

    def forecast(self, name):
        if name in self.cache:
            return self.cache[name]
        if name == "ensemble__ets_branches":
            result = .5 * self.forecast("direct__ets_additive") + .5 * self.forecast("branches__ets_additive")
            self.notes[name] = {"equal_weights": [.5, .5]}
        else:
            architecture, model = name.split("__", 1)
            if architecture == "direct":
                series = self.snapshot.headline.loc[self.common_start:]
                if len(tail_history(series, self.origin, self.window)) < self.snapshot.protocol.min_train:
                    raise ValueError("Historique d'apprentissage commun insuffisant.")
                result, info = predict(series, self.origin, model,
                                       self.snapshot.external, self.window)
                self.notes[name] = info
            elif architecture in {"branches", "countries"}:
                result, paths, history, weights, notes = leaf_paths(
                    self.snapshot, self.origin, architecture, model, self.window)
                self.leaves[name] = (paths, history, weights)
                self.notes[name] = {"leaf_notes": notes, "weights_year": self.origin.year,
                                    "membership_and_weights": "freeze_at_origin"}
            else:
                raise ValueError(f"Architecture inconnue : {name}")
        self.cache[name] = result
        return result


def finite_or_none(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def origin_result(snapshot, origin, models, target_start, target_end, window=None):
    """La construction des prévisions précède toute lecture des réalisations."""
    engine = ForecastEngine(snapshot, origin, window)
    results, failures = {}, {}
    for model in models:
        try:
            results[model] = engine.forecast(model)
        except Exception as exc:
            failures[model] = f"{type(exc).__name__}: {exc}"
    records, leaves = [], []
    for model in models:
        for h in range(1, 4):
            target = origin + h
            if not target_start <= target <= target_end:
                continue
            actual = float(snapshot.headline.loc[target])
            denominator = float(snapshot.headline.loc[target - 12])
            value = float(results[model].loc[target]) if model in results else None
            yoy = 100 * (value / denominator - 1) if value is not None else None
            actual_yoy = 100 * (actual / denominator - 1)
            note = engine.notes.get(model, {})
            fallback_count = sum(bool(n.get("fallback")) for n in note.get("leaf_notes", []))
            records.append({"origin": str(origin), "target": str(target), "horizon": h,
                            "model": model, "status": "ok" if value is not None else "failed",
                            "predicted_index": value, "actual_index": actual,
                            "predicted_yoy": yoy, "actual_yoy": actual_yoy,
                            "error_pp": yoy - actual_yoy if yoy is not None else None,
                            "fallback_count": fallback_count,
                            "crosses_january": origin.year != target.year,
                            "failure": failures.get(model)})
    for model, (paths, _, _) in engine.leaves.items():
        for target in paths.index:
            if not target_start <= target <= target_end:
                continue
            for geo, component in paths.columns:
                leaves.append({"origin": str(origin), "target": str(target),
                               "horizon": target.ordinal - origin.ordinal,
                               "model": model, "geo": geo, "component": component,
                               "predicted_index": finite_or_none(paths.loc[target, (geo, component)]),
                               "actual_index": finite_or_none(snapshot.series(geo, component).loc[target])})
    return {"records": records, "leaves": leaves, "notes": engine.notes}


def metrics(frame):
    """Les échecs rendent une ligne inéligible, au lieu de supprimer des dates."""
    rows = []
    for (model, h), group in frame.groupby(["model", "horizon"], sort=True):
        ok = group["status"].eq("ok") & np.isfinite(group["error_pp"])
        eligible = bool(ok.all())
        e = group.loc[ok, "error_pp"]
        rows.append({"model": model, "horizon": int(h), "expected": len(group),
                     "succeeded": int(ok.sum()), "failures": int((~ok).sum()),
                     "eligible": eligible,
                     "mae_pp": float(e.abs().mean()) if eligible else None,
                     "rmse_pp": float(np.sqrt((e**2).mean())) if eligible else None,
                     "bias_pp": float(e.mean()) if eligible else None,
                     "origins_with_fallback": int(group["fallback_count"].gt(0).sum())})
    return pd.DataFrame(rows).sort_values(["horizon", "rmse_pp"])


def validate_grid(frame, models, start, end):
    expected = pd.MultiIndex.from_product(
        [models, [1, 2, 3], pd.period_range(start, end, freq="M").astype(str)],
        names=["model", "horizon", "target"])
    received = pd.MultiIndex.from_frame(frame[["model", "horizon", "target"]])
    if received.has_duplicates or len(received) != len(expected) or len(expected.difference(received)):
        raise ValueError("La grille de prévisions attendue est incomplète ou dupliquée.")


def period_label(target):
    year = int(target[:4])
    return ("2014-2019" if year <= 2019 else "2020" if year == 2020 else
            "2021-2023" if year <= 2023 else "2024-validation" if target <= "2024-09" else "test-final")


def save_analysis(frame, destination):
    scores = metrics(frame)
    scores.to_csv(destination / "metrics.csv", index=False)
    for column, values in [("period", frame["target"].map(period_label)),
                           ("crosses_january", frame["crosses_january"])]:
        groups = []
        for key, g in frame.assign(**{column: values}).groupby(column):
            groups.append(metrics(g).assign(**{column: key}))
        pd.concat(groups).to_csv(destination / f"metrics_by_{column}.csv", index=False)
    worst = frame.loc[frame["status"].eq("ok") & frame["horizon"].eq(3)].copy()
    worst["absolute_error"] = worst["error_pp"].abs()
    worst.sort_values("absolute_error", ascending=False).groupby("model").head(5).to_csv(
        destination / "worst_errors_h3.csv", index=False)
    rolling = []
    for (name, h), g in frame.groupby(["model", "horizon"]):
        g = g.sort_values("target")
        r = np.sqrt(g["error_pp"].pow(2).rolling(12, min_periods=12).mean())
        rolling.extend({"model": name, "horizon": int(h), "target": t, "rolling_rmse_pp": finite_or_none(v)}
                       for t, v in zip(g["target"], r))
    pd.DataFrame(rolling).to_csv(destination / "rolling_rmse.csv", index=False)
    frame.loc[frame["status"].ne("ok")].to_csv(destination / "fit_failures.csv", index=False)


def run_comparison(snapshot, phase="validation", models=None, window=None, restart=False):
    p = snapshot.protocol
    models = list(models or MODELS)
    if len(set(models)) != len(models) or any(m not in MODELS for m in models):
        raise ValueError("Liste de modèles invalide.")
    if snapshot.external is None and "direct__ridge_ppi" in models:
        raise ValueError("Les PPI sont absents. Préparer prepare_external.py, ou passer explicitement --without-ppi avant init.")
    if window is not None and window < p.min_train:
        raise ValueError("Fenêtre insuffisante pour ce protocole.")
    if phase == "validation":
        start, end = month(p.validation_start), month(p.validation_end)
    elif phase == "test":
        start, end = month(p.test_start), month(p.test_end)
    else:
        raise ValueError("Phase inconnue.")
    configuration = {"phase": phase, "targets": [str(start), str(end)],
                     "models": models, "window": window,
                     "inputs": stable_hash(snapshot.identity),
                     "numerics": numerical_fingerprint()}
    signature = stable_hash(configuration)
    pointer = lab_root() / f"latest_{phase}{'_120' if window == 120 else ''}.json"
    if pointer.exists() and not restart:
        previous = read_json(pointer)
        if previous["signature"] != signature:
            raise ValueError("Configuration différente. Pour une nouvelle validation utiliser --restart; le test final reste lié au gel.")
        destination = lab_root() / previous["directory"]
        if (destination / "_SUCCESS").exists():
            print(f"Résultat identique déjà terminé : {destination}", flush=True)
            return destination
    else:
        stamp = datetime.now(UTC).strftime(f"{phase}_%Y%m%dT%H%M%S_%fZ")
        destination = lab_root() / "runs" / stamp
        destination.mkdir(parents=True)
        write_json(pointer, {"signature": signature, "directory": str(destination.relative_to(lab_root()))})
    cache = destination / "origins"
    cache.mkdir(exist_ok=True)
    write_json(destination / "experiment.json", {**configuration, "signature": signature,
               "status": "running", "evaluation_type": "retrospective_fixed_vintage",
               "release_calendar": "not_reconstructed", "started_at": utcnow()})
    origins = sorted({t - h for t in pd.period_range(start, end, freq="M") for h in [1, 2, 3]})
    records, leaves = [], []
    for i, origin in enumerate(origins):
        path = cache / f"{origin}.json"
        if path.exists():
            result = read_json(path)
        else:
            result = origin_result(snapshot, origin, models, start, end, window)
            write_json(path, result)
        records.extend(result["records"])
        leaves.extend(result["leaves"])
        if i == 0 or (i + 1) % 6 == 0 or i + 1 == len(origins):
            print(f"{phase} {i + 1}/{len(origins)} : origine {origin}", flush=True)
    frame = pd.DataFrame(records)
    validate_grid(frame, models, start, end)
    frame.to_parquet(destination / "predictions.parquet", index=False)
    pd.DataFrame(leaves).to_parquet(destination / "leaf_predictions.parquet", index=False)
    save_analysis(frame, destination)
    write_json(destination / "experiment.json", {**configuration, "signature": signature,
               "status": "completed", "completed_at": utcnow(),
               "evaluation_type": "retrospective_fixed_vintage",
               "release_calendar": "not_reconstructed",
               "failed_forecasts": int(frame["status"].ne("ok").sum()),
               "prediction_sha256": digest(destination / "predictions.parquet"),
               "leaf_prediction_sha256": digest(destination / "leaf_predictions.parquet")})
    (destination / "_SUCCESS").write_text("completed\n")
    print(metrics(frame).query("horizon == 3").to_string(index=False), flush=True)
    return destination


def load_run(path=None, phase="validation", window=None):
    if path is None:
        pointer = lab_root() / f"latest_{phase}{'_120' if window == 120 else ''}.json"
        path = lab_root() / read_json(pointer)["directory"]
    path = Path(path).resolve()
    meta = read_json(path / "experiment.json")
    if not (path / "_SUCCESS").exists() or meta["status"] != "completed":
        raise ValueError("Cette comparaison n'est pas terminée.")
    for file, key in [("predictions.parquet", "prediction_sha256"),
                      ("leaf_predictions.parquet", "leaf_prediction_sha256")]:
        if digest(path / file) != meta[key]:
            raise ValueError("Un résultat de comparaison a été modifié.")
    frame = pd.read_parquet(path / "predictions.parquet")
    validate_grid(frame, meta["models"], *meta["targets"])
    return path, meta, frame
