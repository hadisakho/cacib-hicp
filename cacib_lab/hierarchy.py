"""Agrégation de partitions non chevauchantes et réconciliation positive."""

import numpy as np
import pandas as pd
from .data import BRANCHES, DIVISIONS, month, tail_history
from .models import predict


def weights_at(snapshot, year, architecture):
    """Poids d'une année observée; aucune recherche dans une année future."""
    if year not in snapshot.item_weights.index:
        raise ValueError(f"Poids absents en {year}.")
    if architecture == "branches":
        keys = [("EA", c) for c in BRANCHES]
        w = snapshot.item_weights.loc[year].reindex(keys).astype(float)
        if not np.isfinite(w).all() or w.lt(0).any() or abs(w.sum() - 1000) > .2:
            raise ValueError("Les cinq branches ne constituent pas un panier complet.")
        return w / w.sum()
    if architecture != "countries":
        raise ValueError("Architecture d'agrégation inconnue.")
    if year not in snapshot.country_weights.index:
        raise ValueError("Poids géographiques absents.")
    country = snapshot.country_weights.loc[year]
    members = []
    for geo, entry in snapshot.identity["membership"].items():
        if entry[:4] == "1999" or year >= int(entry[:4]):
            members.append(geo)
    cw = country.reindex(members)
    if not np.isfinite(cw).all() or cw.le(0).any() or abs(cw.sum() - 1000) > .2:
        raise ValueError(f"Poids pays incomplets ou incompatibles en {year}.")
    values = {}
    for geo in members:
        iw = snapshot.item_weights.loc[year].reindex([(geo, c) for c in DIVISIONS])
        if not np.isfinite(iw).all() or iw.lt(0).any() or abs(iw.sum() - 1000) > .2:
            raise ValueError(f"Poids COICOP incomplets : {geo}, {year}.")
        for key, value in iw.items():
            values[key] = float(cw[geo] / cw.sum() * value / iw.sum())
    result = pd.Series(values, dtype=float)
    result.index = pd.MultiIndex.from_tuples(result.index, names=["geo", "component"])
    return result


def linked_history(snapshot, origin, columns):
    """Vue de chaînage avec report du passé, distincte de l'histoire modélisée.

    Un indice de décembre absent peut utiliser la dernière valeur positive
    antérieure. Chaque remplacement effectivement utilisé est journalisé.
    """
    x = snapshot.levels.loc[:month(origin)].reindex(columns=columns).copy()
    good = x.where(np.isfinite(x) & x.gt(0))
    return good.ffill(), good


def aggregate(history, forecasts, weights, headline_history):
    """Poids fixes à l'origine, même au franchissement de janvier.

    Zéro poids : une série manquante n'a aucune contribution. Tout poids
    positif exige une base et une trajectoire positives; pas de renormalisation
    d'un panier amputé.
    """
    origin = headline_history.index[-1]
    future = pd.period_range(origin + 1, periods=len(forecasts), freq="M")
    if not forecasts.index.equals(future) or not 1 <= len(future) <= 3:
        raise ValueError("Trois mois futurs consécutifs au maximum sont attendus.")
    if not weights.index.is_unique or not np.isfinite(weights).all() or weights.lt(0).any():
        raise ValueError("Poids invalides.")
    if abs(weights.sum() - 1) > 1e-8:
        raise ValueError("Le panier normalisé doit sommer à 1.")
    w = weights.loc[weights.gt(0)]
    x = pd.concat([history.reindex(columns=w.index), forecasts.reindex(columns=w.index)])
    if not x.index.is_unique:
        raise ValueError("Historique et prévisions se chevauchent.")
    out = headline_history.to_dict()
    for target in future:
        base = month(f"{target.year - 1}-12")
        if base not in x.index or base not in out:
            raise ValueError("Base de décembre absente.")
        a, b = x.loc[target], x.loc[base]
        if not np.isfinite(a).all() or not np.isfinite(b).all() or a.le(0).any() or b.le(0).any():
            raise ValueError("Une série à poids positif n'a pas d'indice utilisable.")
        out[target] = float(out[base]) * float((a / b * w).sum())
    return pd.Series([out[t] for t in future], index=future, dtype=float)


def leaf_paths(snapshot, origin, architecture, model, window=None):
    origin = month(origin)
    w = weights_at(snapshot, origin.year, architecture)
    # Les pays non encore membres sont aussi prévus à partir de leur seul passé,
    # pour disposer d'erreurs de calibration après leur entrée. Leur poids est nul.
    keys = ([("EA", c) for c in BRANCHES] if architecture == "branches" else
            [(g, c) for g in sorted(snapshot.identity["membership"]) for c in DIVISIONS])
    columns = pd.MultiIndex.from_tuples(keys, names=["geo", "component"])
    future = pd.period_range(origin + 1, periods=3, freq="M")
    paths = pd.DataFrame(index=future, columns=columns, dtype=float)
    notes = []
    common_start = None
    if architecture == "branches":
        starts = [snapshot.series("EA", c).where(lambda s: s.gt(0)).first_valid_index()
                  for c in ["TOTAL", *BRANCHES]]
        common_start = max(starts)
    for geo, component in keys:
        key = (geo, component)
        try:
            series = snapshot.series(*key)
            if common_start is not None:
                series = series.loc[common_start:]
            predicted, info = predict(series, origin, model,
                                      window=window, allow_fallback=True)
            paths[key] = predicted
        except ValueError as exc:
            if w.get(key, 0) > 0:
                raise
            # Une cellule inactive sans historique reste absente, jamais inventée.
            info = {"requested": model, "used": None, "fallback": True,
                    "reason": str(exc), "inactive": True}
        if info.get("fallback") or info.get("warnings"):
            notes.append({"geo": geo, "component": component, **info})
    linked, original = linked_history(snapshot, origin, columns)
    bases = {month(f"{t.year - 1}-12") for t in future if month(f"{t.year - 1}-12") <= origin}
    for base in sorted(bases):
        for key in w.loc[w.gt(0)].index:
            if base not in linked.index or not np.isfinite(linked.loc[base, key]):
                raise ValueError(f"Aucune base de décembre passée : {key}, {base}.")
            if not np.isfinite(original.loc[base, key]):
                notes.append({"geo": key[0], "component": key[1], "fallback": True,
                              "used": "carry_past_December_base", "base": str(base)})
    headline = snapshot.headline.loc[:origin]
    top = aggregate(linked, paths, w, headline)
    return top, paths, linked, w, notes


def reconcile(history, paths, weights, headline_history, target):
    """Facteur multiplicatif commun, calculé mois par mois. Indices positifs."""
    adjusted = paths.copy()
    factors = []
    for i, t in enumerate(paths.index):
        current = aggregate(history, adjusted.iloc[:i + 1], weights, headline_history).loc[t]
        factor = float(target.loc[t] / current)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Réconciliation non positive.")
        adjusted.loc[t] *= factor
        factors.append(factor)
    rebuilt = aggregate(history, adjusted, weights, headline_history)
    np.testing.assert_allclose(rebuilt.to_numpy(), target.to_numpy(), rtol=1e-10, atol=1e-10)
    return adjusted, factors


def historical_diagnostics(snapshot):
    """Contrôle descriptif limité à la validation. Aucun modèle n'est ajusté."""
    rows = []
    start = max(month("2001-01"), snapshot.levels.index[0] + 12)
    for t in pd.period_range(start, month(snapshot.protocol.validation_end), freq="M"):
        base = month(f"{t.year - 1}-12")
        for architecture in ["branches", "countries"]:
            row = {"month": str(t), "architecture": architecture}
            try:
                w = weights_at(snapshot, t.year, architecture)
                positive = w.loc[w.gt(0)]
                a = snapshot.levels.loc[t].reindex(positive.index)
                b = snapshot.levels.loc[base].reindex(positive.index)
                if not np.isfinite(a).all() or not np.isfinite(b).all() or a.le(0).any() or b.le(0).any():
                    raise ValueError("Indices manquants/non positifs : reconstruction stricte indisponible.")
                value = float(snapshot.headline.loc[base] * (a / b * positive).sum())
                row.update(status="ok", official=float(snapshot.headline.loc[t]),
                           reconstructed=value, error_index=value-float(snapshot.headline.loc[t]))
            except (ValueError, KeyError) as exc:
                row.update(status="unavailable", reason=str(exc))
            rows.append(row)
    return pd.DataFrame(rows)
