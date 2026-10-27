"""Données auditées, extraction figée et protocole temporel explicite.

Les valeurs sources sont conservées. Seules les vues utilisées par les modèles
masquent les indices non positifs; aucun remplissage global n'est effectué.
"""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BRANCHES = ["SERV", "IGD_NNRG", "NRG", "FOOD_P", "FOOD_NP"]
DIVISIONS = [f"CP{i:02d}" for i in range(1, 14)]
DIVISION_LABELS = dict(zip(DIVISIONS, [
    "Alimentation et boissons non alcoolisées", "Alcool et tabac",
    "Habillement et chaussures", "Logement, eau, électricité et combustibles",
    "Équipement et entretien du logement", "Santé", "Transport",
    "Information et communication", "Loisirs, sport et culture", "Éducation",
    "Restaurants et hébergement", "Assurances et services financiers",
    "Soins personnels, protection sociale et divers",
]))


@dataclass(frozen=True)
class Protocol:
    validation_start: str = "2014-01"
    validation_end: str = "2024-09"
    test_start: str = "2024-10"
    test_end: str = "2026-09"
    as_of: str = "2026-09"
    horizon: int = 3
    min_train: int = 120
    seed: int = 1729
    min_calibration: int = 36
    interval_draws: int = 2000
    ppi_lag: int = 2

    def __post_init__(self):
        a, b, c, d, e = map(month, [self.validation_start, self.validation_end,
                                  self.test_start, self.test_end, self.as_of])
        if not (a <= b and b + 1 == c and c <= d <= e):
            raise ValueError("Les périodes de validation et de test se chevauchent ou sont incohérentes.")
        if self.horizon != 3 or self.min_train < 120 or self.ppi_lag < 2:
            raise ValueError("Ce protocole impose h=3, au moins 120 mois et un retard PPI >=2.")


def month(value):
    return pd.Period(value, freq="M")


def project_root():
    if os.environ.get("CACIB_PROJECT_ROOT"):
        return Path(os.environ["CACIB_PROJECT_ROOT"]).resolve()
    import config
    return Path(config.PROJECT_ROOT).resolve()


def lab_root():
    return project_root() / "reports" / "lab"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Écriture atomique; un NaN ne devient jamais un JSON non standard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def utcnow():
    return datetime.now(UTC).isoformat()


def numerical_fingerprint():
    """Le gel porte sur le calcul, sans bloquer les ajustements visuels."""
    folder = Path(__file__).parent
    files = [folder / n for n in ["data.py", "models.py", "hierarchy.py",
                                   "evaluation.py", "uncertainty.py", "workflow.py"]]
    import baselines, forecast_models, ridge_model
    files += [Path(m.__file__) for m in (baselines, forecast_models, ridge_model)]
    versions = {}
    for name in ["numpy", "pandas", "scikit-learn", "statsmodels", "pyarrow"]:
        versions[name] = importlib.metadata.version(name)
    return {"code": {p.name: digest(p) for p in files}, "versions": versions}


@dataclass
class Snapshot:
    levels: pd.DataFrame        # colonnes MultiIndex (geo, component)
    item_weights: pd.DataFrame  # index année, mêmes colonnes
    country_weights: pd.DataFrame
    external: pd.DataFrame | None
    protocol: Protocol
    identity: dict
    labels: dict

    def series(self, geo, component):
        key = (geo, component)
        if key not in self.levels.columns:
            return pd.Series(np.nan, index=self.levels.index, dtype=float)
        return self.levels[key].astype(float)

    @property
    def headline(self):
        return self.series("EA", "TOTAL")


def initialize(panel=None, external=None, protocol=None):
    """Fige une seule collecte. Une relance identique conserve le même verrou."""
    protocol = protocol or Protocol()
    root = project_root()
    lock_path = lab_root() / "inputs.json"
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock["protocol"] != asdict(protocol):
            raise ValueError("Un protocole différent existe déjà. Conserver cette expérience et utiliser un autre projet.")
        if panel and Path(panel).resolve() != (root / lock["panel"]).resolve():
            raise ValueError("Une autre collecte est déjà figée.")
        if external and Path(external).resolve() != (root / lock["external"]).resolve():
            raise ValueError("Une autre collecte externe est déjà figée.")
        verify_inputs(lock)
        return lock
    if panel is None:
        candidates = sorted((root / "data/processed/panel").glob("*/_SUCCESS"), reverse=True)
        candidates = [p.parent for p in candidates
                      if read_json(p.parent / "audit.json").get("integrity_status") == "passed"]
        if not candidates:
            raise FileNotFoundError("Exécuter prepare_panel.py avant la suite.")
        panel = candidates[0]
    panel = Path(panel).resolve()
    audit = read_json(panel / "audit.json")
    if not (panel / "_SUCCESS").is_file() or audit.get("integrity_status") != "passed":
        raise ValueError("Le panel sélectionné n'a pas passé l'audit.")
    manifest = Path(audit["source_manifest"])
    # Les anciennes métadonnées peuvent conserver le chemin d'une autre machine.
    if not manifest.exists():
        manifest = root / "data/raw/collections" / f"panel_{audit['collection_id']}.json"
    meta = read_json(manifest)
    if meta.get("end_period") != protocol.as_of or meta.get("index_unit") != "I25":
        raise ValueError("La collecte doit correspondre à la coupure figée et à l'unité I25.")
    if external is None:
        candidates = sorted((root / "data/processed/external").glob("*/_SUCCESS"), reverse=True)
        external = next((p.parent for p in candidates
                         if read_json(p.parent / "audit.json").get("status") == "passed"), None)
    paths = [panel / "audit.json", panel / "country_weights.parquet",
             *sorted((panel / "indices").glob("*.parquet")),
             *sorted((panel / "item_weights").glob("*.parquet"))]
    if external is not None:
        external = Path(external).resolve()
        ea = read_json(external / "audit.json")
        if ea.get("status") != "passed" or ea.get("lag_months") != protocol.ppi_lag:
            raise ValueError("Audit externe ou décalage PPI incompatible.")
        if not (external / "_SUCCESS").exists():
            raise ValueError("Préparation externe inachevée.")
        paths += [external / "audit.json", external / "features.parquet"]
    lock = {"created_at": utcnow(), "protocol": asdict(protocol),
            "panel": os.path.relpath(panel, root),
            "external": os.path.relpath(external, root) if external else None,
            "collection_id": audit["collection_id"],
            "membership": meta["euro_area_entry"],
            "sha256": {os.path.relpath(p, root): digest(p) for p in paths}}
    write_json(lock_path, lock)
    return lock


def verify_inputs(lock):
    for relative, expected in lock["sha256"].items():
        path = project_root() / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"L'extraction figée a été modifiée : {relative}")


def load_snapshot():
    lock = read_json(lab_root() / "inputs.json")
    verify_inputs(lock)
    protocol = Protocol(**lock["protocol"])
    panel = project_root() / lock["panel"]
    components = ["TOTAL", *DIVISIONS, *BRANCHES]
    levels, weights, labels = [], [], {}
    for path in sorted((panel / "indices").glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["unit"].eq("I25") & frame["freq"].eq("M")
                          & frame["coicop18"].isin(components)].copy()
        if frame.duplicated(["time", "geo", "coicop18"]).any():
            raise ValueError(f"Doublons dans {path.name}")
        if "geo_label" in frame and len(frame):
            labels[path.stem] = str(frame["geo_label"].iloc[0])
        frame["month"] = pd.PeriodIndex(frame["time"], freq="M")
        levels.append(frame.pivot(index="month", columns=["geo", "coicop18"], values="value"))
        wf = pd.read_parquet(panel / "item_weights" / path.name)
        wf = wf.loc[wf["coicop18"].isin(components)]
        if wf.duplicated(["year", "geo", "coicop18"]).any():
            raise ValueError(f"Poids dupliqués : {path.name}")
        weights.append(wf.pivot(index="year", columns=["geo", "coicop18"], values="value"))
    x = pd.concat(levels, axis=1).sort_index().astype(float)
    x = x.reindex(pd.period_range(x.index.min(), month(protocol.as_of), freq="M"))
    x.columns.names = ["geo", "component"]
    w = pd.concat(weights, axis=1).sort_index().astype(float)
    w.columns.names = x.columns.names
    cw = pd.read_parquet(panel / "country_weights.parquet")
    if cw.duplicated(["year", "geo"]).any():
        raise ValueError("Poids pays dupliqués.")
    cw = cw.pivot(index="year", columns="geo", values="value").astype(float)
    ext = None
    if lock["external"]:
        ext = pd.read_parquet(project_root() / lock["external"] / "features.parquet")
        ext.index = pd.PeriodIndex(ext.index, freq="M")
        if ext.index.has_duplicates or not ext.index.is_monotonic_increasing:
            raise ValueError("Dates externes invalides.")
        ext = ext.loc[:protocol.as_of].astype(float)
    return Snapshot(x, w, cw, ext, protocol, lock, labels)


def tail_history(series, origin, window=None):
    """Dernier segment observé continu; les trous ne sont pas imputés."""
    x = series.loc[:month(origin)].copy()
    if window is not None:
        x = x.iloc[-window:]
    valid = np.isfinite(x) & x.gt(0)
    bad = np.flatnonzero(~valid.to_numpy())
    if len(bad):
        x = x.iloc[bad[-1] + 1:]
    return x
