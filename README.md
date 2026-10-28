# Euro area inflation (HICP): forecast and shock pass-through

An end-to-end pipeline on the **Eurostat API** (HICP, ECOICOP ver.2, index 2025 = 100):

- **Data:**
  - monthly indices for the headline, the 5 main aggregates, the 13 COICOP ver.2 divisions, core and HICPxT;
  - for the euro area and its 21 member countries, with item weights and country weights;
  - cleaned, quality-reported and checked against the published numbers.
- **Forecast:** headline YoY 3 months ahead, evaluated out-of-sample against naive benchmarks.
- **Shock simulator:** price shocks on one or more components (main aggregates or COICOP divisions) and/or one or more countries, passed through exactly to euro area headline inflation (direct effects). The scenario baseline is the central forecast. Indirect energy effects are an optional, exploratory extra, and oil shocks are sized with market data.
- **Interactive dashboard** (Streamlit) and an analysis notebook.

The euro area is the evolving composition (Eurostat geo `EA`: EA11-1999 … EA21-2026).

## Results (Eurostat data to September 2026)

| | Oct 2026 | Nov 2026 | Dec 2026 |
|---|---|---|---|
| Headline YoY forecast, recommended model | 3.96% | 4.05% | **4.14%** (80%: 3.6–5.1%) |

**Recommended model:** the equal-weight average of four models from two families:
- ETS on the headline;
- ETS on the 5 main aggregates (re-aggregated);
- a seasonal ARX on the aggregates;
- a seasonal median on the aggregates.

**Selection :** The central forecast averages four models, each with a weight of 25%. The model set was chosen after
reviewing the historical results, so the backtest comparisons are exploratory. On the common evaluation period, equal 
weights perform about as well as inverse-MSE weighting.

| Window | Recommended | No change, RMSE (pp) |
|---|---|---|
| Full, 2013–2026 (163 forecasts) | **0.77** (DM p = 0.05) | 0.87 |
| Validation, 2014-01 → 2024-09 | 0.76 | 0.94 |
| Test, 2024-10 → 2026-09 | 0.93 | 0.53 |
| Calm, 2013–2019 | 0.93 | 0.46 |
| Surge, 2020–2023 | 0.72 | 1.42 |


**Selection rules replayed month by month** (`outputs/selection.csv`). Each rule chooses at every forecast date using only the errors already published then, so the choice itself is out-of-sample. A rule starts once 24 errors are known (targets from May 2015 on).

| Rule (3-month RMSE / no change) | Full | Validation | Test | Calm | Surge | Model switches |
|---|---|---|---|---|---|---|
| **Equal weights, fixed (rule used)** | **0.754** | 0.742 | 0.926 | 0.885 | 0.722 | 0 |
| Inverse-MSE weights | 0.755 | 0.743 | 0.938 | 0.885 | 0.722 | 0 |
| Best past RMSE | 0.794 | 0.772 | 1.090 | 0.884 | 0.754 | 1 |
| Best past RMSE, last 36 months | 0.793 | 0.781 | 0.959 | 0.848 | 0.769 | 9 |
| Best past RMSE, any model (incl. average, naive) | 0.812 | 0.800 | 0.986 | 1.007 | 0.767 | 6 |

- Every rule that picks one model from its past errors doesn't perform better than fixed equal weights. Ranking differs 
in subperiods.
- No model leads in every window: ETS headline in the surge, the seasonal median on the test window, ETS bottom-up in calm years. That is when picking one model fails and averaging is the standard choice.
- Picking the lowest validation RMSE would choose ETS headline (0.711 pp, just ahead of the average's 0.713), which then scores 1.09 × no change on the test window.

**Results and  limitations:**
- The gain over the no-change forecast is borderline significant: Diebold–Mariano p ≈ 0.05, Hansen SPA p ≈ 0.09 after 
allowing for the model search.
- It is concentrated in volatile periods.
- Volatility-scaled 80% intervals cover 76% of outcomes in real time.
- 24 overlapping three-month forecast errors, which limits the precision of model comparisons.
- A fully independent confirmation would need outcomes not yet published.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync --locked                          # create .venv with pinned versions
uv run pytest                             # 31 tests
uv run python -m hicp.pipeline --refresh  # Eurostat download, cleaning, checks, backtest, forecast (~2 min)
uv run streamlit run app.py               # dashboard at http://localhost:8501
uv run jupyter lab notebooks/hicp_analysis.ipynb
```

Other options:
- without `--refresh`, the pipeline reuses the data cached in `data/`;
- `--quick` skips the backtest;
- `--demo` runs offline on synthetic data and writes to `outputs/demo/`.

**PyCharm:**
1. *Settings → Python Interpreter → Add Local Interpreter → Select existing → `.venv/bin/python`*.
2. Run configurations: module `hicp.pipeline`; module `streamlit` with parameters `run app.py`; pytest on `tests`.

**Inspect a Eurostat dataset:**

```bash
uv run python -m hicp.eurostat describe prc_hicp_minr --search energy
```

## Project layout

```
config.yaml          datasets, codes, aggregates, divisions, model settings, evaluation windows
app.py               Streamlit dashboard (reads data/processed and outputs/, simulates shocks live)
hicp/
  eurostat.py        Eurostat API client (JSON-stat 2.0), raw-response cache, `describe` command
  data.py            download -> cleaning -> panels (indices, item weights, country weights) -> parquet
  quality.py         cleaning rules and per-series quality report
  aggregation.py     chain-linking, exact contributions, effective weights
  validation.py      checks that the engine reproduces the published numbers (both breakdowns)
  forecast.py        base-effect identity, models, recommended average, backtest
  evaluation.py      accuracy by window, Diebold-Mariano, Hansen SPA, intervals, coverage,
                     selection rules replayed month by month
  passthrough.py     local projections of the aggregates on energy prices
  shocks.py          shock shapes, exact re-aggregation, indirect effects, fuel-price elasticity
  market.py          Brent and EUR/USD (FRED) and the fuels share of energy (Eurostat weights)
  plots.py           Plotly charts (one colour per component everywhere)
  synthetic.py       offline demo data with the same structure
  pipeline.py        `python -m hicp.pipeline`
notebooks/hicp_analysis.ipynb
tests/
data/processed/, outputs/   committed so the hosted dashboard works without network
```

## Method

**Data.** The Eurostat dissemination API (JSON-stat):
- `prc_hicp_minr`: monthly indices, unit I25.
- `prc_hicp_iw`: item weights, `statinfo=IW`.
- `prc_hicp_cw`: country weights, `statinfo=COWEA`, i.e. the euro area with evolving composition.

The COICOP dimension is `coicop18`. Every raw response is cached with its download time.

**Cleaning.**
- Exact duplicates are dropped; conflicting duplicates stop the pipeline.
- Non-positive or non-finite index values are removed.
- Negative weights stop the pipeline.
- Nothing is interpolated.
- Eurostat status flags (`d` definition differs, `e` estimated, `b` break, `u` low reliability, `p` provisional) are kept and reported.
- `outputs/data_quality.csv` gives, per series: coverage, interior gaps, values removed and flags.

**Aggregation.** HICP is chain-linked every year with a December overlap. For month t of year y, with D = Dec(y−1): `I_t / I_D = Σ_i w_i^y · I_i,t / I_i,D`.
- The headline YoY is therefore not a weighted mean of component YoY rates. `aggregation.contributions` gives the exact (Ribe) decomposition.
- The euro area is rebuilt from 21 countries with weights ω = country weight × item weight.
- For both partitions (5 aggregates, 13 divisions), it matches the published rate to within 0.02 pp.

**Forecast.** `π_T+3 = π_T + Σ_k (m_T+k − m_T+k−12)`. The base effects `m_T+k−12` are known, so the models forecast only 3 monthly changes.
- Component paths are re-aggregated and applied to the published index.
- For a year whose weights aren't published yet, the previous year's weights are price-updated.
- Every model is re-estimated each month on the previous 120 months only.

**Evaluation.**
- Forecast origins run monthly from 2012-12; each forecast uses only data available at its origin (unit-tested).
- Diebold–Mariano tests with the Harvey–Leybourne–Newbold correction (3-step errors overlap: 2 autocovariances).
- Hansen SPA test for data snooping.
- Real-time interval coverage.
- The validation/test split of the original protocol is kept as two evaluation windows. Because the model was chosen with the whole sample in view, they are descriptive, not an independent test.
- Selection rules replayed month by month (`evaluation.selection_replay`): each rule only sees errors already published at the time.

**Shocks.** `I_shock = I_base · (1 + s)` at country × component level, for one or more components (main aggregates or COICOP divisions) and one or more countries, re-aggregated exactly.
- **Shapes:** step, ramp, temporary, or rate (the price level grows x% a year faster, compounded, so the component's annual rate becomes (1 + π)(1 + x) − 1: about π + x at low inflation, but 13.3% rather than 13% at π = 10%, x = 3).
- **Rule of thumb:** a permanent x% rise on basket share ω adds ≈ ω·x pp to headline YoY for 12 months, then drops out.
- **Baseline.** For the 3 forecast months it equals the central forecast: every series is scaled by one common factor from each forecast month on, so the aggregate hits the forecast exactly and relative prices are unchanged. A scenario without shocks therefore returns the forecast. Later months (up to 24) continue each series' usual seasonal pattern and recent momentum. They are a conditional assumption, not a forecast, since accuracy is only tested at 1–3 months. Impacts (shocked − baseline) barely depend on this path.
- **Direct effects (default)** are exact accounting: no estimated coefficient.
- **Indirect effects (optional, exploratory; main aggregates, energy shocks only):** energy → food, goods and services, from local projections. Both the start month t and the outcome month t+h lie in 1997–2020, so the 2021–23 surge is never used, at any horizon. The regressor is the observed energy price change, not an identified shock. The dashboard shows the range obtained with each component's response at the ends of its 90% band.
- **Oil shocks are sized from data:**
  - Brent and EUR/USD are 30-day averages from FRED;
  - the pump-price elasticity is crude cost / (crude + excise + margin) per litre, with excise and margin as stated assumptions;
  - the fuels share of energy comes from Eurostat's weights (`FUEL` / `NRG`).
- **Gas & electricity shocks are sized the same way:** a rise in their prices moves each country's energy HICP by that rise × their weight in energy (`CP0451` + `CP0452` / `NRG`, about 52% in DE and 53% in IT).

## Assumptions and limitations

- The backtest uses the latest data vintage. HICP is rarely revised, but the ver.2 back series did not exist in real time.
- The forecast is univariate: no energy futures and no flash estimate. This is the main avenue for improvement, since energy explains 1.7 pp of September 2026's 3.8%.
- The model was chosen after looking at the whole sample, so the backtest comparison is exploratory. Picking the lowest validation RMSE would have chosen ETS headline, which scores 1.09 × no change on the test window; the replay above shows that equal weights beat every written-down selection rule. Independent confirmation needs outcomes published after the choice. Expect gains of a few percent in calm periods.
- Shocks:
  - weights after the last published year are held fixed;
  - indirect pass-through is the same in every country and is not modelled for divisions;
  - there are no second-round (wage) effects;
  - the pass-through regressor is not an identified shock (upgrade: LP-IV).

## Refresh and deployment

- **Monthly refresh** `.github/workflows/refresh.yml` re-runs the pipeline on the 20th of each month and commits `data/processed/` and `outputs/`. It needs *Settings → Actions → General → Workflow permissions → Read and write*.
- **Hosting** On [Streamlit Community Cloud](https://share.streamlit.io): *Create app*, then this repository, branch `main`, file `app.py`. The dependencies come from `requirements.txt`, which you regenerate with `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.
