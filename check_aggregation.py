"""Diagnostic rétrospectif du chaînage des cinq composantes HICP.

Usage : uv run python check_aggregation.py
Ne consulte pas les cibles des 24 mois réservés au test final.
Les poids sont ex post ; ce script n'est pas un moteur de prévision.
"""

import hashlib
import json
from datetime import UTC, datetime

import config
import numpy as np
import pandas as pd
from prepare_panel import select_manifest

BRANCHES = ["SERV", "IGD_NNRG", "NRG", "FOOD_P", "FOOD_NP"]


def reconstruct(levels, weights, start, end):
    """Chaînage continu et diagnostic annuel avec ancre officielle distincte."""
    start, end = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    if start.month != 1 or start > end:
        raise ValueError("Commencer en janvier, avant la fin de la période.")
    if levels.index.has_duplicates or weights.index.has_duplicates:
        raise ValueError("Les dates ou années doivent être uniques.")
    required = pd.period_range(start - 1, end, freq="M")
    x = levels.reindex(index=required, columns=["TOTAL", *BRANCHES])
    w = weights.reindex(index=range(start.year, end.year + 1), columns=BRANCHES)
    if not np.isfinite(x.to_numpy(dtype=float)).all() or (x <= 0).any().any():
        raise ValueError("Indices absents, non finis ou non positifs.")
    if not np.isfinite(w.to_numpy(dtype=float)).all() or (w < 0).any().any():
        raise ValueError("Poids absents, non finis ou négatifs.")
    sums = w.sum(axis=1)
    if not np.allclose(sums, 1000, atol=0.2, rtol=0):
        raise ValueError("La somme des cinq poids diffère de 1000 au-delà de 0,2.")
    chain = {start - 1: float(x.loc[start - 1, "TOTAL"])}
    rows = []
    for month in pd.period_range(start, end, freq="M"):
        base = pd.Period(f"{month.year - 1}-12", freq="M")
        relative = x.loc[month, BRANCHES] / x.loc[base, BRANCHES]
        # Normalisation par la somme publiée, conformément à l'agrégation.
        factor = float((relative * w.loc[month.year]).sum() / sums.loc[month.year])
        chain[month] = chain[base] * factor
        anchored = float(x.loc[base, "TOTAL"]) * factor
        official = float(x.loc[month, "TOTAL"])
        rows.append(
            {
                "month": str(month),
                "official": official,
                "chain_reconstructed": chain[month],
                "annual_anchor_reconstructed": anchored,
                "chain_error_index": chain[month] - official,
                "annual_anchor_error_index": anchored - official,
                "published_weight_sum": float(sums.loc[month.year]),
            }
        )
    result = pd.DataFrame(rows).set_index("month")
    result.index = pd.PeriodIndex(result.index, freq="M")
    for column in ["official", "chain_reconstructed"]:
        result[column + "_yoy"] = 100 * (result[column] / result[column].shift(12) - 1)
    result["yoy_error_pp"] = result["chain_reconstructed_yoy"] - result["official_yoy"]
    return result


def main():
    manifest_path, manifest = select_manifest()
    directory = config.PROCESSED_DATA_DIR / "panel" / manifest["collection_id"]
    audit_path = directory / "audit.json"
    if not (directory / "_SUCCESS").is_file():
        raise RuntimeError("Exécuter prepare_panel.py avant ce diagnostic.")
    if json.loads(audit_path.read_text())["integrity_status"] != "passed":
        raise ValueError("Le panel ne passe pas les contrôles bloquants.")
    indices_path = directory / "indices" / f"{config.EURO_AREA}.parquet"
    weights_path = directory / "item_weights" / f"{config.EURO_AREA}.parquet"
    frame = pd.read_parquet(indices_path)
    frame = frame.loc[
        frame["coicop18"].isin(["TOTAL", *BRANCHES])
        & frame["geo"].eq(config.EURO_AREA)
        & frame["unit"].eq(config.INDEX_UNIT)
        & frame["freq"].eq(config.FREQUENCY)
    ].copy()
    frame["month"] = pd.PeriodIndex(frame["time"], freq="M")
    levels = frame.pivot(index="month", columns="coicop18", values="value").astype(
        float
    )
    weight_frame = pd.read_parquet(weights_path)
    weight_frame = weight_frame.loc[weight_frame["coicop18"].isin(BRANCHES)]
    weights = weight_frame.pivot(
        index="year", columns="coicop18", values="value"
    ).astype(float)
    # Première année complète pour les cinq poids dans cette extraction.
    start = pd.Period("2001-01", freq="M")
    end = pd.Period(config.END_PERIOD, freq="M") - 24
    result = reconstruct(levels, weights, start, end)
    summary = {
        "collection_id": manifest["collection_id"],
        "start": str(start),
        "end": str(end),
        "months": len(result),
        "method": "December unchaining, weighted mean, continuous rechaining",
        "initial_anchor": "official December 2000 only",
        "retrospective_weights": True,
        "source_manifest": str(manifest_path),
        "source_sha256": {
            p.name + "_" + p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [indices_path, weights_path]
        },
        "script_sha256": hashlib.sha256(
            __import__("pathlib").Path(__file__).read_bytes()
        ).hexdigest(),
    }
    for name in ["chain_error_index", "annual_anchor_error_index", "yoy_error_pp"]:
        errors = result[name].dropna()
        summary[name] = {
            "n": len(errors),
            "mae": float(errors.abs().mean()),
            "rmse": float(np.sqrt((errors**2).mean())),
            "max_abs": float(errors.abs().max()),
        }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = config.PROJECT_ROOT / "reports" / "aggregation" / stamp
    destination.mkdir(parents=True)
    result.to_csv(destination / "reconstruction.csv", index_label="month")
    by_year = result.groupby(result.index.year)[
        ["chain_error_index", "annual_anchor_error_index", "yoy_error_pp"]
    ].agg(["mean", lambda s: s.abs().max()])
    by_year.to_csv(destination / "errors_by_year.csv")
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Rapports :", destination)
    print(
        "Diagnostic terminé ; une faible erreur n'est pas un test de performance prédictive."
    )


if __name__ == "__main__":
    main()
