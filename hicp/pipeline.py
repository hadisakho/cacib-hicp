"""End-to-end run: Eurostat data -> cleaning and checks -> backtest -> forecast ->
pass-through -> market inputs. Writes the tables the dashboard reads to outputs/.

    python -m hicp.pipeline                 # Eurostat API (cached in data/)
    python -m hicp.pipeline --refresh       # re-download after a new monthly release
    python -m hicp.pipeline --quick         # skip the backtest (about 1 minute)
    python -m hicp.pipeline --demo          # synthetic data, offline (writes outputs/demo/)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import pandas as pd

from .config import load_config
from .data import load_or_download
from .evaluation import accuracy_by_window, interval_bands, realtime_coverage, spa_test
from .forecast import BENCHMARKS, MEMBERS, NO_CHANGE, RECOMMENDED, ForecastSetup, backtest, final_forecast
from .market import fuels_share_of_energy, market_snapshot
from .passthrough import energy_passthrough
from .synthetic import make_demo_data
from .validation import run_checks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--demo", action="store_true", help="synthetic data (offline)")
    ap.add_argument("--refresh", action="store_true", help="re-download the data")
    ap.add_argument("--quick", action="store_true", help="skip the backtest")
    args = ap.parse_args(argv)

    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)  # noqa: E731
    cfg = load_config(args.config)
    out = cfg.output_dir / ("demo" if args.demo else "")
    out.mkdir(parents=True, exist_ok=True)

    data = make_demo_data(cfg) if args.demo else load_or_download(cfg, refresh=args.refresh)
    log(f"data: {data.source}, {data.indices.shape[1]} series, last month {data.last_month}, cleaning {data.cleaning}")
    if data.quality is not None:
        data.quality.to_csv(out / "data_quality.csv", index=False)

    checks = run_checks(data)
    checks.to_csv(out / "checks.csv", index=False)
    log(f"checks: {int(checks['passed'].sum())}/{len(checks)} passed")
    if not checks["passed"].all():
        print(checks.loc[~checks["passed"], ["group", "check", "value", "tolerance", "unit"]].to_string(index=False))

    f = cfg["forecast"]
    setup = ForecastSetup(data.area_components(), data.area_component_weights(), data.headline(),
                          h=f["horizon"], window=f["window"], seasonal_years=f["seasonal_years"])
    if not args.quick:
        bt = backtest(setup.forecasters(), data.headline(), f["backtest_start"])
        bt.to_csv(out / "backtest.csv", index=False)
        acc = accuracy_by_window(bt, setup.h, cfg.windows)
        acc.to_csv(out / "accuracy.csv", index=False)
        candidates = [m for m in bt["model"].unique() if m not in BENCHMARKS]
        pd.DataFrame([{"window": w, "spa_p_value": spa_test(bt, setup.h, candidates, NO_CHANGE, a, b)}
                      for w, (a, b) in cfg.windows.items()]).to_csv(out / "spa.csv", index=False)
        pd.DataFrame([realtime_coverage(bt, RECOMMENDED, h, 0.8, scale)
                      for h in range(1, setup.h + 1) for scale in (False, True)]).to_csv(out / "coverage.csv", index=False)
        full = acc[acc["window"] == next(iter(cfg.windows))].set_index("model")
        log(f"backtest: {RECOMMENDED} RMSE {full.loc[RECOMMENDED, 'RMSE']:.3f} pp, "
            f"{full.loc[RECOMMENDED, 'rel. RMSE']:.2f} x no change (DM p {full.loc[RECOMMENDED, 'DM p-value']:.3f})")

    ff = final_forecast(setup)
    table = ff["models"].rename_axis("target").reset_index()
    table.insert(1, "h", range(1, len(table) + 1))
    if (out / "backtest.csv").exists():
        table = table.merge(interval_bands(pd.read_csv(out / "backtest.csv"), RECOMMENDED).reset_index(), on="h")
    table["target"] = table["target"].astype(str)
    table.to_csv(out / "forecast.csv", index=False)
    log("forecast: " + ", ".join(f"{t} {v:.2f}%" for t, v in zip(table["target"], table[RECOMMENDED])))

    p = cfg["passthrough"]
    lp = energy_passthrough(data.area_components(), energy=cfg.energy, horizons=p["horizons"], lags=p["lags"],
                            sample=tuple(p["sample"]) if p.get("sample") else None)
    pd.concat({c: lp[c] for c in data.components}, names=["component"]).to_csv(out / "passthrough.csv")
    lp["ratio"].to_csv(out / "passthrough_ratio.csv")
    log("pass-through estimated")

    try:
        market = {} if args.demo else market_snapshot(cfg, data, refresh=args.refresh)
    except Exception as exc:  # optional input: the dashboard falls back to its defaults
        warnings.warn(f"market data unavailable ({exc})")
        market = {}
    if "fuels_share_of_energy" not in market:
        share, src = fuels_share_of_energy(data, cfg["shocks"]["fuels_share_fallback"])
        market.update(fuels_share_of_energy=share, fuels_share_source=src)
    (out / "market.json").write_text(json.dumps(market, indent=2), encoding="utf-8")
    log(f"market inputs: {market}")

    meta = {"source": data.source, "data_vintage": data.vintage, "last_month": str(data.last_month),
            "recommended": RECOMMENDED, "members": MEMBERS,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"done - outputs in {out}")


if __name__ == "__main__":
    main()
