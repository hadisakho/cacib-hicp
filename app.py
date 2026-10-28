"""Euro area inflation dashboard.

    streamlit run app.py

Reads data/processed/ and outputs/ (written by `python -m hicp.pipeline`); only the shock
simulator computes live.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from hicp import plots
from hicp.aggregation import contributions, effective_weights, yoy
from hicp.config import load_config
from hicp.data import load
from hicp.evaluation import EQUAL_WEIGHTS
from hicp.forecast import BENCHMARKS, NO_CHANGE, RECOMMENDED, base_effect_decomposition
from hicp.market import share_of_energy
from hicp.passthrough import ratio_bands
from hicp.shocks import Shock, fuel_price_elasticity, simulate

st.set_page_config(page_title="Euro area inflation", page_icon="📈", layout="wide")
OUTPUTS = ["forecast.csv", "backtest.csv", "accuracy.csv", "spa.csv", "coverage.csv", "checks.csv",
           "passthrough.csv", "metadata.json"]
FLAGS = {"p": "provisional", "e": "estimated", "d": "definition differs", "b": "break in series",
         "u": "low reliability", "c": "confidential", "s": "Eurostat estimate", "f": "forecast"}
AGGREGATES, DIVISIONS = "Main aggregates", "COICOP ver.2 divisions"


def show(fig, where=st) -> None:
    """Render a chart with the project style (set explicitly: Streamlit overrides template values)."""
    fig.update_layout(paper_bgcolor=plots.SURFACE, plot_bgcolor=plots.SURFACE,
                      font=dict(family=plots.FONT, color=plots.INK))
    where.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


@st.cache_resource
def load_everything():
    cfg = load_config(Path(__file__).parent / "config.yaml")
    out = cfg.output_dir
    missing = [f for f in OUTPUTS if not (out / f).exists()]
    if missing or not (cfg.processed_dir / "metadata.json").exists():
        return None
    data = load(cfg)
    bt = pd.read_csv(out / "backtest.csv")
    for c in ("origin", "target"):
        bt[c] = pd.PeriodIndex(bt[c], freq="M")
    lp = {c: g.drop(columns="component").set_index("h")
          for c, g in pd.read_csv(out / "passthrough.csv").groupby("component")}
    read_json = lambda f: json.loads((out / f).read_text()) if (out / f).exists() else {}  # noqa: E731
    return {
        "cfg": cfg, "data": data, "bt": bt,
        "forecast": pd.read_csv(out / "forecast.csv"), "accuracy": pd.read_csv(out / "accuracy.csv"),
        "spa": pd.read_csv(out / "spa.csv"), "coverage": pd.read_csv(out / "coverage.csv"),
        "checks": pd.read_csv(out / "checks.csv"),
        "quality": pd.read_csv(out / "data_quality.csv") if (out / "data_quality.csv").exists() else None,
        "selection": pd.read_csv(out / "selection.csv", index_col=0) if (out / "selection.csv").exists() else None,
        "lp": lp, "ratios": ratio_bands(lp, cfg.energy),  # central / low / high pass-through ratios
        "meta": read_json("metadata.json"), "market": read_json("market.json"),
        "contrib": {name: contributions(data.area_breakdown(codes), data.area_weights(codes))
                    for name, codes in breakdowns(cfg).items()},
    }


def breakdowns(cfg) -> dict[str, list[str]]:
    return {AGGREGATES: cfg.components, **({DIVISIONS: cfg.divisions} if cfg.divisions else {})}


@st.cache_data(show_spinner="Simulating...")
def run_shocks(specs: tuple, breakdown: str, ratio: str | None, horizon: int, anchor: tuple):
    """ratio: None (direct effects only) or "central" / "low" / "high" pass-through ratios.
    anchor: central forecast as ((month, YoY %), ...), reproduced by the baseline."""
    R = load_everything()
    cfg = R["cfg"]
    return simulate(R["data"], [Shock(*s) for s in specs], horizon=horizon, energy=cfg.energy,
                    passthrough_ratio=R["ratios"][ratio] if ratio else None, codes=breakdowns(cfg)[breakdown],
                    anchor=pd.Series(dict(anchor)))


R = load_everything()
if R is None:
    st.title("Euro area inflation")
    st.error("No results yet. Run the pipeline first: `uv run python -m hicp.pipeline` "
             "(downloads from the Eurostat API and writes data/processed/ and outputs/).")
    st.stop()

cfg, data, fc = R["cfg"], R["data"], R["forecast"]
labels = {**data.labels, plots.OTHER: "Other divisions"}
head, T = data.headline(), data.last_month
rate = yoy(head)
fc_targets = pd.PeriodIndex(fc["target"], freq="M")
fc_yoy = pd.Series(fc[RECOMMENDED].to_numpy(), index=fc_targets)
largest_divisions = list(data.area_weights(data.divisions).iloc[-1].sort_values(ascending=False).index[:7]) \
    if data.divisions else []

st.title("Euro area inflation")
st.caption(f"HICP, euro area (evolving composition EA11-1999 … EA21-2026), ECOICOP ver.2, index 2025 = 100 · "
           f"data to {T.strftime('%B %Y')} · source: {R['meta'].get('source', 'Eurostat API')} · "
           f"updated {R['meta'].get('run_at', '')[:10]}")

tab_over, tab_fc, tab_shock, tab_method = st.tabs(["Overview", "Forecast", "Shock simulator", "Method & data"])

# ---- Overview -----------------------------------------------------------------------
with tab_over:
    contrib_agg = R["contrib"][AGGREGATES]
    c = st.columns(4)
    c[0].metric(f"Headline, {T.strftime('%b %Y')}", f"{rate.loc[T]:.1f}%",
                f"{rate.loc[T] - rate.loc[T - 1]:+.1f} pp vs previous month", delta_color="off")
    core = data.series(data.area, data.codes["core"]) if "core" in data.codes else None
    if core is not None:
        core_rate = yoy(core)
        c[1].metric("Core (ex. energy, food, alcohol, tobacco)", f"{core_rate.loc[T]:.1f}%",
                    f"{core_rate.loc[T] - core_rate.loc[T - 1]:+.1f} pp", delta_color="off")
    last = fc.iloc[-1]
    c[2].metric(f"Forecast, {fc_targets[-1].strftime('%b %Y')}", f"{last[RECOMMENDED]:.1f}%",
                f"80% range {last[RECOMMENDED] + last['lo80']:.1f}–{last[RECOMMENDED] + last['hi80']:.1f}%",
                delta_color="off")
    c[3].metric("Energy contribution", f"{contrib_agg.loc[T, cfg.energy]:+.2f} pp",
                f"{contrib_agg.loc[T, cfg.energy] - contrib_agg.loc[T - 12, cfg.energy]:+.2f} pp over 12 months",
                delta_color="off")

    show(plots.headline_chart(head, core))
    view = st.segmented_control("Breakdown", list(R["contrib"]), default=AGGREGATES, key="contrib_view")
    contrib = R["contrib"][view or AGGREGATES]
    if (view or AGGREGATES) == DIVISIONS:
        contrib = plots.keep_and_other(contrib, largest_divisions)
    show(plots.contributions_chart(contrib, head, labels))
    show(plots.country_heatmap(data))

# ---- Forecast -----------------------------------------------------------------------
with tab_fc:
    bands = fc.set_index("h")[[c for c in fc.columns if c[:2] in ("lo", "hi")]]
    show(plots.fan_chart(head, fc_yoy, bands))

    implied = pd.concat([head, pd.Series(head.reindex(fc_targets - 12).to_numpy() * (1 + fc_yoy.to_numpy() / 100),
                                         index=fc_targets)])
    left, right = st.columns([3, 2])
    show(plots.base_effects_chart(base_effect_decomposition(implied, T, len(fc_targets))), left)
    with right:
        st.markdown("**Forecasts by model** (headline YoY, %)")
        models = fc.set_index("target")[[c for c in fc.columns if c not in ("target", "h") and c[:2] not in ("lo", "hi")]]
        st.dataframe(models.T.round(2), width="stretch")
        st.caption("The recommended model is the equal-weight average of the four models above it. Intervals: "
                   "empirical quantiles of past out-of-sample errors, widened when recent errors are larger than usual.")

    st.subheader("Does it beat a naive forecast?")
    acc = R["accuracy"]
    window = st.segmented_control("Evaluation window", list(cfg.windows), default=next(iter(cfg.windows)))
    table = acc[acc["window"] == (window or next(iter(cfg.windows)))].drop(columns="window")
    show(plots.accuracy_chart(table, RECOMMENDED, BENCHMARKS))
    st.dataframe(table.set_index("model").round(3), width="stretch")
    with st.expander("Overfitting and interval checks"):
        spa = R["spa"].set_index("window")["spa_p_value"]
        at5, at10 = [w for w, p in spa.items() if p < 0.05], [w for w, p in spa.items() if 0.05 <= p < 0.10]
        verdict = (f"Rejected at 5% only in: {', '.join(at5)}" if at5 else "Not rejected at 5% in any window") + (
            f"; at 10% also in: {', '.join(at10)}" if at10 else "")
        st.markdown(
            "- **Data snooping (Hansen SPA test).** Null hypothesis: no candidate beats the no-change forecast, "
            "allowing for having searched over all of them. p-values: "
            + ", ".join(f"{w} {p:.3f}" for w, p in spa.items()) + f". {verdict}.\n"
            "- **Real-time 80% interval coverage** (target 0.80): "
            + ", ".join(f"h={int(r.h)} {'scaled' if r.scaled else 'plain'} {r.coverage:.2f}" for r in R["coverage"].itertuples())
            + f".\n- Every model is re-estimated each month on the previous {cfg['forecast']['window']} months only; "
            "the combination has no estimated weights.")
        st.markdown("**How the model was chosen (read the comparison above as exploratory).** The four models and "
                    "the decision to average them were settled after seeing the whole sample, test window included, "
                    "so these scores are not an independent confirmation. The rule itself, equal weights on all "
                    "candidates with nothing estimated, uses no outcome data. Below, written-down selection rules are "
                    "replayed month by month, each seeing only the errors already published at the time "
                    "(RMSE relative to no change; below 1 beats it):")
        if R["selection"] is not None:
            sel = R["selection"]
            st.dataframe(sel.style.format(precision=3).format(precision=0, subset=["n", "model switches"])
                         .apply(lambda r: ["font-weight: bold" if r.name == EQUAL_WEIGHTS else "" for _ in r], axis=1),
                         width="stretch")
        windows = set(acc["window"])
        if {"validation", "test"} <= windows:
            cands = acc[~acc["model"].isin(BENCHMARKS)]
            val = cands[cands["window"] == "validation"].sort_values("RMSE").iloc[0]["model"]
            test = acc[(acc["window"] == "test") & (acc["model"] == val)].iloc[0]["rel. RMSE"]
            st.caption(f"Picking the lowest validation RMSE would choose {val}, which then scores {test:.2f} × no "
                       "change on the test window. A fully independent test needs outcomes not yet published "
                       "when the model was chosen.")
    show(plots.track_record_chart(R["bt"], head, [RECOMMENDED, NO_CHANGE]))


# ---- Shock simulator ----------------------------------------------------------------
with tab_shock:
    months = [str(T + k) for k in range(1, 13)]
    presets = ["Oil price shock (fuels channel)", "German services prices +3%",
               "Gas & electricity +15% in DE and IT", "Transport +10% in France (COICOP CP07)", "Custom"]
    controls, results = st.columns([1, 2.6], gap="large")
    with controls:
        preset = st.selectbox("Scenario", presets)
        specs: list[tuple] = []
        breakdown, indirect = AGGREGATES, False
        if preset == presets[0]:
            m = R["market"]
            move = st.slider("Brent price change (%)", -50, 100, 30, 5)
            with st.expander("Pass-through assumptions"):
                brent = st.number_input("Brent before the shock ($/bbl)", value=float(m.get("brent_usd", 80.0)))
                fx = st.number_input("USD per EUR", value=float(m.get("usd_per_eur", 1.10)), format="%.4f")
                excise = st.number_input("Excise (€/litre)", value=float(cfg["shocks"]["excise_eur_per_litre"]))
                margin = st.number_input("Refining & distribution margin (€/litre)",
                                         value=float(cfg["shocks"]["margin_eur_per_litre"]))
            eps = fuel_price_elasticity(brent, fx, excise, margin)
            share = float(m.get("fuels_share_of_energy", cfg["shocks"]["fuels_share_fallback"]))
            size = move * eps * share
            st.caption(f"Brent {brent:.0f} $/bbl and EUR/USD {fx:.3f} ({m.get('source', 'defaults')}, "
                       f"{m.get('as_of', 'n/a')}). Pump-price elasticity {eps:.2f} → fuels {move * eps:+.1f}% → "
                       f"energy HICP {size:+.1f}% (fuels = {share:.1%} of energy, {m.get('fuels_share_source', '')}).")
            specs.append((cfg.energy, size, months[0], None, "ramp", 2, 6))
        elif preset == presets[1]:
            specs.append(("SERV", 3.0, months[0], ("DE",), "ramp", 6, 6))
            st.caption("A direct 3% rise in German services prices, phased in over 6 months (e.g. the price effect "
                       "assumed after a wage settlement). No wage → price mechanism is estimated.")
        elif preset == presets[2]:  # electricity & gas are part of energy: size the energy shock by their weight
            move, codes, parts = 15.0, [data.codes[k] for k in ("electricity", "gas") if k in data.codes], []
            for geo in ("DE", "IT"):
                share, src = share_of_energy(data, codes, geo, cfg["shocks"]["elec_gas_share_fallback"], cfg.energy)
                specs.append((cfg.energy, move * share, months[0], (geo,), "temporary", 1, 6))
                parts.append(f"{geo} {move * share:+.1f}% (electricity & gas = {share:.0%} of energy)")
            st.caption(f"Electricity & gas prices +{move:.0f}% for 6 months → energy HICP " + ", ".join(parts)
                       + f"; {src}.")
        elif preset == presets[3]:
            breakdown = DIVISIONS
            specs.append(("CP07", 10.0, months[0], ("FR",), "step", 1, 6))
        else:
            breakdown = st.radio("Breakdown", list(breakdowns(cfg)), horizontal=True,
                                 help="Main aggregates allow optional indirect effects of energy shocks; "
                                      "COICOP divisions are direct effects only.")
            names = {labels.get(k, k): k for k in breakdowns(cfg)[breakdown]}
            chosen = st.multiselect("Components (one or more)", list(names), default=list(names)[:1])
            countries = st.multiselect("Countries (empty = whole euro area)", data.countries)
            size = st.number_input("Price level change (%)", value=10.0, step=1.0)
            shape = st.selectbox("Shape", ["step", "ramp", "temporary", "rate"],
                                 help="step: permanent level shift · ramp: phased in · temporary: reverses · "
                                      "rate: the price level grows x% a year faster, compounded, so the annual "
                                      "rate becomes (1 + π)(1 + x) − 1, e.g. 2% → 5.06% or 10% → 13.3% for x = 3")
            start = st.selectbox("Start", months)
            ramp = st.number_input("Ramp length (months)", 1, 24, 6) if shape == "ramp" else 6
            duration = st.number_input("Duration (months)", 1, 24, 6) if shape == "temporary" else 6
            specs += [(names[n], float(size), start, tuple(countries) or None, shape, int(ramp), int(duration))
                      for n in chosen]
        if breakdown == AGGREGATES and any(spec[0] == cfg.energy for spec in specs):  # energy shocks only
            indirect = st.toggle("Add indirect effects (exploratory)", value=False,
                                 help="Energy → food, goods and services, from local projections on "
                                      f"{'–'.join(m[:4] for m in cfg['passthrough']['sample'])} "
                                      "(observed energy price changes, not identified shocks; same pass-through "
                                      "in every country). Shown with a range from the coefficients' 90% bands.")
        st.caption("Rule of thumb: a permanent x% rise on a basket share ω adds ≈ ω·x pp to headline "
                   "inflation for 12 months, then drops out.")

    with results:
        if not specs:
            st.info("Choose at least one component.")
        else:
            anchor = tuple((str(t), float(v)) for t, v in fc_yoy.items())
            run = lambda ratio: run_shocks(tuple(specs), breakdown, ratio, cfg["shocks"]["horizon"], anchor)  # noqa: E731
            res = run("central" if indirect else None)
            show(plots.shock_chart(res, labels, preset))
            if indirect:
                peak = res.delta["total"].abs().idxmax()
                ends = sorted(run(r).delta.loc[peak, "total"] for r in ("low", "high"))
                direct = res.delta.loc[peak, "direct"]
                st.caption(f"Sensitivity to the estimated coefficients: with each component's response at the "
                           f"low or high end of its 90% band, the impact in {peak.strftime('%b %Y')} ranges from "
                           f"{ends[0]:+.2f} to {ends[1]:+.2f} pp (central {res.delta.loc[peak, 'total']:+.2f} pp, "
                           f"direct effects alone {direct:+.2f} pp).")
            show(plots.shock_country_chart(res))
            table = pd.concat([res.yoy, res.delta.add_prefix("impact ")], axis=1)
            table.index = table.index.astype(str)
            st.download_button("Download scenario (CSV)", table.round(4).to_csv(), "scenario.csv", "text/csv")
    st.caption(f"Baseline: equals the central forecast ({RECOMMENDED.split(' (')[0]}) for "
               f"{fc_targets[0].strftime('%b')}–{fc_targets[-1].strftime('%b %Y')}, so a scenario without shocks "
               "reproduces it exactly. After that, each series continues with its usual seasonal pattern and recent "
               "momentum: a conditional assumption, not a forecast (forecast accuracy is only tested at 1–3 months). "
               "Impacts (shocked − baseline) barely depend on the baseline. Direct effects are exact accounting; "
               "indirect effects are optional and exploratory. Weights after the last published year held fixed; "
               "no second-round (wage) effects.")

# ---- Method & data ------------------------------------------------------------------
with tab_method:
    st.subheader("Data checks")
    st.caption("The aggregation engine reproduces Eurostat's published numbers for both breakdowns.")
    checks = R["checks"]
    blank_text = {c: "" for c in checks.columns if checks[c].dtype == object}  # numbers stay numeric
    st.dataframe(checks.fillna(blank_text).set_index(["group", "check"]).round(4), width="stretch")

    st.subheader("Data cleaning and quality")
    flags = data.flags
    left, right = st.columns(2)
    with left:
        st.markdown("**Cleaning rules**: exact duplicates dropped; conflicting duplicates stop the pipeline; "
                    "non-positive or non-finite index values removed; negative weights stop the pipeline; "
                    "nothing is interpolated.")
        st.dataframe(pd.Series(data.cleaning, name="count").to_frame(), width="stretch")
    with right:
        if flags is not None and len(flags):
            counts = flags["flag"].value_counts().rename(index=lambda f: f"{f} ({FLAGS.get(f, '?')})")
            st.markdown("**Eurostat status flags** (kept and reported, not used to drop data)")
            st.dataframe(counts.rename("observations").to_frame(), width="stretch")
            recent = flags[flags["time"] >= T - 11]
            st.caption(f"In the last 12 months: {len(recent)} flagged observations "
                       f"({', '.join(f'{k}: {v}' for k, v in recent['flag'].value_counts().items()) or 'none'}).")
    if R["quality"] is not None:
        with st.expander("Per-series coverage (first/last month, gaps, values removed, flags)"):
            st.dataframe(R["quality"], width="stretch", hide_index=True)

    omega = effective_weights(data.item_weights, data.country_weights, data.countries, data.components)
    year = int(omega.index[-1])
    show(plots.weights_chart(omega.loc[year], labels, year))
    show(plots.passthrough_chart(R["lp"], labels, data.components, energy=cfg.energy,
                                 sample=tuple(cfg["passthrough"]["sample"])))
    es = cfg["eurostat"]["datasets"]
    st.subheader("Method")
    st.markdown(f"""
- **Data.** Eurostat dissemination API: `{es['index']}` (monthly indices, unit I25),
  `{es['item_weights']}` (item weights), `{es['country_weights']}` (country weights, euro area with
  evolving composition). Headline, 5 main aggregates, 13 COICOP ver.2 divisions, core and HICPxT for the
  euro area and 21 countries.
- **Aggregation.** HICP is chain-linked every year with a December overlap. Exact contributions (Ribe)
  add up to the published rate; the euro area is rebuilt from 21 countries with weights = country
  weight × item weight, for both breakdowns.
- **Forecast.** Headline YoY in 3 months = today's rate + 3 new monthly changes − 3 known base
  effects. Recommended model: equal-weight average of ETS on the headline, ETS on each main aggregate,
  a seasonal ARX on each and a seasonal median on each, re-estimated every month on the last
  {cfg['forecast']['window']} months.
- **Evaluation.** Every forecast uses only data available at its origin. Diebold-Mariano tests
  (HLN correction), Hansen SPA test for data snooping, real-time interval coverage, sensitivity to settings.
- **Shocks.** Price-level shocks on any main aggregate or COICOP division, any countries; exact
  re-aggregation (direct effects, the default). Oil shocks sized with Brent and EUR/USD (FRED), the fuels
  weight (Eurostat) and the pump-price tax structure. The baseline reproduces the central forecast over
  its 3 months; later months are a conditional path.
- **Indirect effects (exploratory).** Energy → food, goods and services from local projections; both the
  origin month and the outcome month t+h lie in {cfg['passthrough']['sample'][0][:4]}–{cfg['passthrough']['sample'][1][:4]}, so the 2021-23 surge is
  never used. Regressor = observed energy price change, not an identified shock; shown with a range
  from the coefficients' 90% bands.
- **Limits.** Univariate forecast: no energy futures or flash estimate. Gains over the no-change
  forecast are concentrated in volatile periods and borderline significant.
""")
