"""Chocs explicites sur des niveaux IPCH et réagrégation exacte.

Un choc de prix de 10 % n'est pas une hausse de 10 points de l'inflation.
Les poids et les autres trajectoires sont maintenus conditionnellement fixes.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import month
from .hierarchy import aggregate


@dataclass(frozen=True)
class Shock:
    countries: tuple[str, ...]
    components: tuple[str, ...]
    magnitude_pct: float
    start: str
    kind: str = "permanent"
    duration: int = 1

    def __post_init__(self):
        if self.kind not in {"permanent", "temporary", "monthly"}:
            raise ValueError("Type de choc inconnu.")
        if not self.countries or not self.components:
            raise ValueError("Sélectionner au moins un pays et une catégorie.")
        if not np.isfinite(self.magnitude_pct) or self.magnitude_pct <= -100:
            raise ValueError("Le choc doit être fini et supérieur à -100 %.")
        if not isinstance(self.duration, (int, np.integer)) or isinstance(self.duration, bool) or self.duration < 1:
            raise ValueError("La durée doit être un entier positif.")
        month(self.start)


def apply_shocks(paths, shocks):
    adjusted = paths.copy()
    occupied = set()
    for shock in shocks:
        start = month(shock.start)
        if start not in paths.index:
            raise ValueError("Le choc doit commencer dans l'horizon prévu.")
        cells = [(g, c) for g in shock.countries for c in shock.components]
        for cell in cells:
            if cell not in paths.columns:
                raise ValueError(f"Cellule inconnue : {cell}.")
            if cell in occupied:
                raise ValueError("Deux chocs ciblent la même cellule. Les combiner explicitement avant simulation.")
            occupied.add(cell)
        amplitude = 1 + shock.magnitude_pct / 100
        for t in paths.index:
            offset = t.ordinal - start.ordinal
            if offset < 0:
                factor = 1.0
            elif shock.kind == "permanent":
                factor = amplitude
            elif shock.kind == "temporary":
                factor = amplitude if offset < shock.duration else 1.0
            else:
                # Après duration hausses mensuelles, le niveau acquis persiste.
                factor = amplitude ** min(offset + 1, shock.duration)
            for cell in cells:
                adjusted.loc[t, cell] *= factor
    active = adjusted.to_numpy()[np.isfinite(adjusted.to_numpy())]
    if len(active) and (active <= 0).any():
        raise ValueError("Le scénario produit un indice non positif.")
    return adjusted


def simulate(history, paths, weights, headline_history, shocks):
    for shock in shocks:
        for g in shock.countries:
            for c in shock.components:
                if weights.get((g, c), 0) <= 0:
                    raise ValueError(f"{g}/{c} n'a pas de poids positif dans le panier de ce scénario.")
    central = aggregate(history, paths, weights, headline_history)
    changed = apply_shocks(paths, shocks)
    shocked = aggregate(history, changed, weights, headline_history)
    rows = []
    for t in paths.index:
        denominator = float(headline_history.loc[t - 12])
        base_yoy = 100 * (central.loc[t] / denominator - 1)
        shock_yoy = 100 * (shocked.loc[t] / denominator - 1)
        rows.append({"target": str(t), "baseline_index": float(central.loc[t]),
                     "scenario_index": float(shocked.loc[t]),
                     "baseline_yoy": float(base_yoy), "scenario_yoy": float(shock_yoy),
                     "impact_pp": float(shock_yoy - base_yoy)})
    return pd.DataFrame(rows)


def decomposition(history, paths, weights, headline_history, shocks):
    """Décomposition séquentielle qui somme exactement, même au changement d'année.

    Avec des interactions de chaînage, l'attribution dépend de l'ordre explicite
    pays/catégorie. Ce n'est pas une attribution causale ni une valeur de Shapley.
    """
    singles = [Shock((g,), (c,), s.magnitude_pct, s.start, s.kind, s.duration)
               for s in shocks for g in sorted(s.countries) for c in sorted(s.components)]
    # Validation complète une seule fois, puis mise à jour incrémentale : le
    # coût reste linéaire dans le nombre de cellules choisies.
    simulate(history, paths, weights, headline_history, shocks)
    adjusted = paths.copy()
    previous = aggregate(history, adjusted, weights, headline_history)
    rows = []
    for single in singles:
        adjusted = apply_shocks(adjusted, [single])
        current = aggregate(history, adjusted, weights, headline_history)
        for t in current.index:
            rows.append({"target": str(t), "geo": single.countries[0],
                         "component": single.components[0],
                         "impact_pp": float(100*(current.loc[t]-previous.loc[t])/headline_history.loc[t-12])})
        previous = current
    return pd.DataFrame(rows, columns=["target", "geo", "component", "impact_pp"])
