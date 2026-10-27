# Euro area inflation (HICP): forecast and shock pass-through

An end-to-end pipeline on the **Eurostat API** (HICP, ECOICOP ver.2, index 2025 = 100):

- **Data:**
  - monthly indices for the headline, the 5 main aggregates, the 13 COICOP ver.2 divisions, core and HICPxT;
  - for the euro area and its 21 member countries, with item weights and country weights;
  - cleaned, quality-reported and checked against the published numbers.
- **Forecast:** headline YoY 3 months ahead, evaluated out-of-sample against naive benchmarks.
- **Shock simulator:** price shocks on one or more components (main aggregates or COICOP divisions) and/or one or more countries, passed through exactly to euro area headline inflation. Indirect energy effects are included, and oil shocks are sized with market data.
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

3-month RMSE relative to the no-change forecast (random walk on inflation); below 1 beats it:

| Window | Recommended | No change, RMSE (pp) |
|---|---|---|
| Full, 2013–2026 (163 forecasts) | **0.77** (DM p = 0.05) | 0.87 |
| Validation, 2014-01 → 2024-09 | 0.76 | 0.94 |
| Test, 2024-10 → 2026-09 | 0.93 | 0.53 |
| Calm, 2013–2019 | 0.93 | 0.46 |
| Surge, 2020–2023 | 0.72 | 1.42 |

**Robust to its settings.** The full-sample ratio stays between 0.765 and 0.774 across estimation windows of 60/96/120/180 months and seasonal windows of 4/6/8 years, and every setting beats the no-change forecast in every window.

**Honest reading:**
- The gain over the no-change forecast is borderline significant: Diebold–Mariano p ≈ 0.05, Hansen SPA p ≈ 0.09 after allowing for the model search.
- It is concentrated in volatile periods.
- Volatility-scaled 80% intervals cover 76% of outcomes in real time.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync --locked                          # create .venv with pinned versions
uv run pytest                             # 27 tests
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
  evaluation.py      accuracy by window, Diebold-Mariano, Hansen SPA, intervals, coverage
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
- The validation/test split of the original protocol is kept as two evaluation windows.

**Shocks.** `I_shock = I_base · (1 + s)` at country × component level, for one or more components (main aggregates or COICOP divisions) and one or more countries, re-aggregated exactly.
- **Shapes:** step, ramp, temporary, or rate (a persistent addition to the component's inflation rate).
- **Rule of thumb:** a permanent x% rise on basket share ω adds ≈ ω·x pp to headline YoY for 12 months, then drops out.
- **Indirect effects** (main aggregates only): energy → food, goods and services, from local projections estimated on 1997–2020. The 2021–23 surge roughly doubles the estimates.
- **Oil shocks are sized from data:**
  - Brent and EUR/USD are 30-day averages from FRED;
  - the pump-price elasticity is crude cost / (crude + excise + margin) per litre, with excise and margin as stated assumptions;
  - the fuels share of energy comes from Eurostat's weights (`FUEL` / `NRG`).

## Assumptions and limitations

- The backtest uses the latest data vintage. HICP is rarely revised, but the ver.2 back series did not exist in real time.
- The forecast is univariate: no energy futures and no flash estimate. This is the main avenue for improvement, since energy explains 1.7 pp of September 2026's 3.8%.
- The model was chosen after looking at the whole sample. Applying the selection rule to validation years only would have picked "ETS bottom-up", which scores 0.97 × no change on the test window. Expect gains of a few percent in calm periods.
- Shocks:
  - weights after the last published year are held fixed;
  - indirect pass-through is the same in every country and is not modelled for divisions;
  - there are no second-round (wage) effects;
  - the pass-through regressor is not an identified shock (upgrade: LP-IV).

## Refresh and deployment

- **Monthly refresh.** `.github/workflows/refresh.yml` re-runs the pipeline on the 20th of each month and commits `data/processed/` and `outputs/`. It needs *Settings → Actions → General → Workflow permissions → Read and write*.
- **Hosting.** On [Streamlit Community Cloud](https://share.streamlit.io): *Create app*, then this repository, branch `main`, file `app.py`. The dependencies come from `requirements.txt`, which you regenerate with `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.
