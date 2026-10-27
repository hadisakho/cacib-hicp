"""Analyse les erreurs de validation, sans réestimer les modèles.

uv run python analyze_errors.py
uv run python analyze_errors.py --run reports/backtests/architectures_...
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from backtest import summarize


def period_label(targets):
    years = targets.str[:4].astype(int)
    return np.select(
        [years <= 2019, years == 2020, years.between(2021, 2023)],
        ["2014-2019", "2020", "2021-2023"],
        default="2024-validation",
    )


def validate_run(directory):
    metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if metadata["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("La comparaison n'est pas terminée.")
    frame = pd.read_parquet(directory / "predictions.parquet")
    start, end = metadata["validation_targets"]
    reserved_start = metadata["reserved_test_targets"][0]
    if (
        not frame["target"].between(start, end).all()
        or not frame["target"].lt(reserved_start).all()
    ):
        raise ValueError(
            "Des cibles sortent de la validation ou empiètent sur le test final."
        )
    names = [
        f"{architecture}__{model}"
        for model in metadata["models"]
        for architecture in ("direct", "bottom_up")
    ]
    targets = pd.period_range(start, end, freq="M").astype(str)
    planned = pd.MultiIndex.from_product(
        [names, (1, 2, 3), targets], names=["model", "horizon", "target"]
    )
    actual = pd.MultiIndex.from_frame(frame[["model", "horizon", "target"]])
    if (
        actual.has_duplicates
        or len(actual) != len(planned)
        or len(planned.difference(actual))
    ):
        raise ValueError(
            "La matrice planifiée de prévisions est incomplète ou dupliquée."
        )
    if not frame["status"].isin(["ok", "failed"]).all():
        raise ValueError("Statut de prévision inconnu.")
    ok = frame["status"].eq("ok")
    if not np.isfinite(
        frame.loc[ok, ["error_pp", "actual_yoy", "predicted_yoy"]].to_numpy()
    ).all():
        raise ValueError("Prévisions déclarées réussies mais non finies.")
    np.testing.assert_allclose(
        frame.loc[ok, "error_pp"],
        frame.loc[ok, "predicted_yoy"] - frame.loc[ok, "actual_yoy"],
    )
    frame["period"] = period_label(frame["target"])
    frame["crosses_january"] = frame["origin"].str[:4].ne(frame["target"].str[:4])
    return frame, metadata


def paired_comparison(frame, models):
    rows = []
    for (period, horizon), group in frame.groupby(["period", "horizon"]):
        for model in models:
            selected = group.loc[
                group["model"].isin([f"direct__{model}", f"bottom_up__{model}"])
            ]
            pivot = selected.pivot(index="target", columns="model", values="error_pp")
            complete = (
                selected["status"].eq("ok").all()
                and np.isfinite(pivot.to_numpy()).all()
            )
            direct = pivot[f"direct__{model}"]
            bottom = pivot[f"bottom_up__{model}"]
            rows.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "model": model,
                    "paired_targets": len(pivot),
                    "eligible": bool(complete),
                    "rmse_bottom_minus_direct_pp": float(
                        np.sqrt((bottom**2).mean()) - np.sqrt((direct**2).mean())
                    )
                    if complete
                    else np.nan,
                    "mae_bottom_minus_direct_pp": float(
                        bottom.abs().mean() - direct.abs().mean()
                    )
                    if complete
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main(run=None):
    if run is None:
        candidates = sorted(
            (config.PROJECT_ROOT / "reports" / "backtests").glob("architectures_*"),
            reverse=True,
        )
        run = next(
            (
                p
                for p in candidates
                if (p / "experiment.json").is_file()
                and json.loads((p / "experiment.json").read_text())[
                    "status"
                ].startswith("completed")
            ),
            None,
        )
        if run is None:
            raise FileNotFoundError("Exécuter compare_architectures.py d'abord.")
    run = Path(run).resolve()
    frame, metadata = validate_run(run)
    output = (
        config.PROJECT_ROOT
        / "reports"
        / "error_analysis"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    scores = summarize(frame)
    scores.to_csv(output / "overall_metrics.csv", index=False)
    periods = pd.concat(
        [summarize(g).assign(period=p) for p, g in frame.groupby("period")],
        ignore_index=True,
    )
    periods.to_csv(output / "metrics_by_period.csv", index=False)
    january = pd.concat(
        [
            summarize(g).assign(crosses_january=k)
            for k, g in frame.groupby("crosses_january")
        ],
        ignore_index=True,
    )
    january.to_csv(output / "metrics_by_january.csv", index=False)
    paired = paired_comparison(frame, metadata["models"])
    paired.to_csv(output / "paired_comparison.csv", index=False)
    worst = (
        frame.loc[frame["status"].eq("ok") & frame["horizon"].eq(3)]
        .assign(absolute_error_pp=lambda x: x["error_pp"].abs())
        .sort_values("absolute_error_pp", ascending=False)
        .groupby("model", sort=False)
        .head(5)
    )
    worst.to_csv(output / "largest_errors_h3.csv", index=False)

    h3 = frame.loc[frame["horizon"].eq(3)].copy()
    rolling_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for name, group in h3.groupby("model"):
        group = group.sort_values("target")
        errors = group["error_pp"].where(group["status"].eq("ok"))
        rolling = errors.pow(2).rolling(12, min_periods=12).mean().pow(0.5)
        dates = pd.PeriodIndex(group["target"], freq="M").to_timestamp()
        axes[0].plot(dates, rolling, label=name)
        rolling_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "target": group["target"].to_numpy(),
                    "rmse_12_months_pp": rolling.to_numpy(),
                }
            )
        )
    axes[0].set_title("Horizon 3 mois — RMSE sur 12 cibles successives")
    axes[0].set_ylabel("Points de pourcentage")
    axes[0].legend(fontsize=8, ncol=2)
    display_scores = scores.loc[scores["horizon"].eq(3)].sort_values("rmse_pp")
    axes[1].barh(display_scores["model"], display_scores["rmse_pp"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMSE de validation — points de pourcentage")
    axes[1].set_title("Scores globaux : dates identiques, échecs exclus du classement")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output / "error_comparison.png", dpi=160)
    plt.close(fig)
    pd.concat(rolling_rows).to_csv(output / "rolling_rmse_h3.csv", index=False)
    provenance = {
        "source_run": str(run),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [run / "experiment.json", run / "predictions.parquet"]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Descriptive validation diagnostics, not a significance test. Periods based on target dates. Negative paired differences favor bottom-up.",
        "reserved_test_scored": False,
    }
    (output / "analysis.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(
        display_scores[["model", "mae_pp", "rmse_pp", "bias_pp", "coverage"]].to_string(
            index=False
        )
    )
    print(
        "\nComparaison par période à h=3 (écart négatif = avantage aux composantes) :"
    )
    print(paired.loc[paired["horizon"].eq(3)].to_string(index=False))
    print("\nRapports :", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    main(parser.parse_args().run)
