"""Plotly charts shared by the notebook (and a future dashboard).

One colour per component everywhere (config order -> categorical slots), recessive
grid, conclusion-style titles, units in every subtitle, no dual axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .aggregation import yoy

# ---- palette ---------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = {"100": "#cde2fb", "200": "#9ec5f4", "300": "#6da7ec", "450": "#2a78d6", "600": "#184f95"}
UP, DOWN, NEUTRAL = "#e34948", "#2a78d6", "#f0efec"   # diverging poles + grey midpoint
TARGET = 2.0  # ECB inflation target, % YoY
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

pio.templates["hicp"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=13, color=INK),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
    title=dict(x=0, xanchor="left", font=dict(size=18)),
    xaxis=dict(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS,
               tickfont=dict(color=INK_2), zeroline=False),
    yaxis=dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, tickfont=dict(color=INK_2),
               zerolinecolor=AXIS, zerolinewidth=1),
    # legend in the bottom margin, below the x axis: it can wrap onto several lines at any
    # width without running into the title (charts without a legend shrink the margin)
    legend=dict(orientation="h", x=0, xanchor="left", y=0, yanchor="bottom", yref="container",
                entrywidth=0.5, entrywidthmode="fraction", font=dict(color=INK_2)),
    hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
    margin=dict(l=60, r=30, t=90, b=130), barcornerradius=4, bargap=0.25,
))
pio.templates.default = "hicp"


OTHER = "OTHER"


def component_colors(components: list[str]) -> dict[str, str]:
    """Fixed colour per code, in the given order; "OTHER" (and anything past 8) is grey."""
    return {c: MUTED if c == OTHER or i >= len(SERIES) else SERIES[i] for i, c in enumerate(components)}


def _fit_legend(fig: go.Figure, columns: int = 2) -> go.Figure:
    """Size the bottom margin to the legend (2 entries per row) so it never covers the axis."""
    n = sum(1 for t in fig.data if t.showlegend is not False and t.name)
    fig.update_layout(margin=dict(b=60 + 22 * -(-n // columns)))
    return fig


def keep_and_other(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Keep some columns and sum the rest into "OTHER" (charts never show more than 8 hues)."""
    rest = [c for c in df.columns if c not in keep]
    out = df[[c for c in keep if c in df.columns]].copy()
    if rest:
        out[OTHER] = df[rest].sum(axis=1)
    return out


def _title(text: str, subtitle: str) -> dict:
    return dict(text=f"{text}<br><span style='font-size:13px;color:{INK_2}'>{subtitle}</span>")


def _ts(index: pd.PeriodIndex) -> pd.DatetimeIndex:
    return index.to_timestamp()


def _target_line(fig: go.Figure, **kw) -> None:
    fig.add_hline(y=TARGET, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text="2% target", annotation_position="top left",
                  annotation_font_color=MUTED, **kw)


# ---- overview --------------------------------------------------------------------

def headline_chart(headline: pd.Series, core: pd.Series | None = None, since: str = "2015-01") -> go.Figure:
    h = yoy(headline).dropna().loc[since:]
    fig = go.Figure()
    fig.add_scatter(x=_ts(h.index), y=h.round(2), name="Headline", line=dict(color=INK, width=2))
    if core is not None:
        c = yoy(core).dropna().loc[since:]
        fig.add_scatter(x=_ts(c.index), y=c.round(2), name="Core (ex. energy, food, alcohol, tobacco)",
                        line=dict(color=SERIES[0], width=2))
    _target_line(fig)
    last = h.index[-1]
    fig.update_layout(title=_title(f"Euro area inflation at {h.iloc[-1]:.1f}% in {last.strftime('%B %Y')}",
                                   "HICP, annual rate of change, %"),
                      hovermode="x unified", yaxis_ticksuffix="%")
    return _fit_legend(fig)


def contributions_chart(contrib: pd.DataFrame, headline: pd.Series, labels: dict[str, str],
                        since: str = "2019-01") -> go.Figure:
    c = contrib.loc[since:]
    h = yoy(headline).reindex(c.index)
    colors = component_colors(list(c.columns))
    fig = go.Figure()
    for comp in c.columns:
        fig.add_bar(x=_ts(c.index), y=c[comp].round(2), name=labels.get(comp, comp),
                    marker=dict(color=colors[comp], line=dict(color=SURFACE, width=1)))
    fig.add_scatter(x=_ts(h.index), y=h.round(2), name="Headline", mode="lines",
                    line=dict(color=INK, width=2))
    top = c.iloc[-1].drop(OTHER, errors="ignore").idxmax()
    fig.update_layout(barmode="relative", hovermode="x unified", bargap=0.15,
                      title=_title(f"Largest contribution: {labels.get(top, top)}, "
                                   f"{c.iloc[-1][top]:.1f} pp of the {h.iloc[-1]:.1f}% headline rate",
                                   "Exact contributions to euro area HICP annual rate (chain-linked), pp"))
    return _fit_legend(fig)


def country_heatmap(data, months: int = 36) -> go.Figure:
    rates = pd.DataFrame({g: yoy(data.headline(g)) for g in data.countries}).iloc[-months:]
    order = rates.iloc[-1].sort_values(ascending=False).index
    z = rates[order].T
    span = 6.0
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy().round(1), x=_ts(z.columns), y=list(order), zmid=TARGET,
        zmin=TARGET - span, zmax=TARGET + span, xgap=1, ygap=1,
        colorscale=[[0, DOWN], [0.5, NEUTRAL], [1, UP]],
        colorbar=dict(title="% YoY", ticksuffix="%", thickness=12),
        hovertemplate="%{y} · %{x|%b %Y}: %{z:.1f}%<extra></extra>"))
    fig.update_layout(margin=dict(b=50), title=_title(f"{order[0]} has the highest rate, {order[-1]} the lowest",
                                   "HICP annual rate by country, % (colour centred on the 2% target, capped at ±6 pp)"),
                      height=620, yaxis=dict(autorange="reversed", showgrid=False))
    return fig


def weights_chart(omega_year: pd.Series, labels: dict[str, str], year: int) -> go.Figure:
    """Each country's share of the euro area basket, split by component (%)."""
    w = 100 * omega_year.unstack("coicop")
    w = w.loc[w.sum(axis=1).sort_values().index]
    colors = component_colors(list(w.columns))
    fig = go.Figure()
    for comp in w.columns:
        fig.add_bar(y=list(w.index), x=w[comp].round(2), name=labels.get(comp, comp), orientation="h",
                    marker=dict(color=colors[comp], line=dict(color=SURFACE, width=1)),
                    hovertemplate="%{y}: %{x:.2f}% of the euro area basket<extra>" + labels.get(comp, comp) + "</extra>")
    big = w.sum(axis=1).sort_values(ascending=False)
    fig.update_layout(barmode="stack", height=640, xaxis_ticksuffix="%", legend_traceorder="normal",
                      title=_title(f"{big.index[0]}, {big.index[1]} and {big.index[2]} make up "
                                   f"{big.iloc[:3].sum():.0f}% of the euro area basket",
                                   f"Effective weights {year}: country weight × item weight, % of euro area HICP"))
    return _fit_legend(fig)


# ---- forecast --------------------------------------------------------------------

def fan_chart(headline: pd.Series, forecast: pd.Series, bands: pd.DataFrame,
              since: str = "2023-01") -> go.Figure:
    """bands: rows = horizon 1..h, columns lo50/hi50/lo80/hi80/lo90/hi90 (error quantiles)."""
    hist = yoy(headline).dropna().loc[since:]
    x_f = _ts(forecast.index)
    fig = go.Figure()
    for lvl, color in ((90, BLUE_RAMP["100"]), (80, BLUE_RAMP["200"]), (50, BLUE_RAMP["300"])):
        lo = forecast.to_numpy() + bands[f"lo{lvl}"].to_numpy()
        hi = forecast.to_numpy() + bands[f"hi{lvl}"].to_numpy()
        xs = [_ts(hist.index[-1:])[0], *x_f]
        y0 = hist.iloc[-1]
        fig.add_scatter(x=[*xs, *xs[::-1]], y=[y0, *hi, *lo[::-1], y0], fill="toself", mode="lines",
                        line=dict(width=0), fillcolor=color, name=f"{lvl}% interval", hoverinfo="skip")
    fig.add_scatter(x=_ts(hist.index), y=hist.round(2), name="Published", line=dict(color=INK, width=2))
    fig.add_scatter(x=[_ts(hist.index[-1:])[0], *x_f], y=[hist.iloc[-1], *forecast.round(2)],
                    name="Forecast", line=dict(color=BLUE_RAMP["600"], width=2, dash="dash"),
                    mode="lines+markers", marker=dict(size=8))
    _target_line(fig)
    end = forecast.index[-1]
    fig.update_layout(hovermode="x unified", yaxis_ticksuffix="%",
                      title=_title(f"We expect headline inflation at {forecast.iloc[-1]:.1f}% in {end.strftime('%B %Y')} "
                                   f"(80%: {forecast.iloc[-1] + bands['lo80'].iloc[-1]:.1f}–"
                                   f"{forecast.iloc[-1] + bands['hi80'].iloc[-1]:.1f}%)",
                                   "Euro area HICP, annual rate %; intervals from out-of-sample forecast errors"))
    return _fit_legend(fig)


def base_effects_chart(decomp: pd.DataFrame) -> go.Figure:
    measure = ["absolute"] + ["relative"] * (len(decomp) - 2) + ["total"]
    vals = decomp["pp"].round(2)
    fig = go.Figure(go.Waterfall(
        x=decomp["term"], y=vals, measure=measure,
        text=[(f"{v:+.2f}" if abs(v) >= 0.005 else "0.00") if m == "relative" else f"{v:.2f}%"
              for v, m in zip(vals, measure)],
        textposition="outside", connector=dict(line=dict(color=AXIS, width=1)),
        increasing=dict(marker=dict(color=UP)), decreasing=dict(marker=dict(color=DOWN)),
        totals=dict(marker=dict(color=MUTED))))
    base = decomp.loc[decomp["kind"] == "base", "pp"].sum()
    new = decomp.loc[decomp["kind"] == "new", "pp"].sum()
    fig.update_layout(showlegend=False, yaxis_ticksuffix="%", margin=dict(b=80),
                      title=_title(f"Base effects contribute {base:+.2f} pp, new price changes {new:+.2f} pp",
                                   "Monthly changes entering (new) and leaving (base) the 12-month window, pp"))
    return fig


def accuracy_chart(table: pd.DataFrame, highlight: str, benchmarks: list[str]) -> go.Figure:
    """RMSE by model: highlighted model in blue, benchmarks grey, others light blue."""
    t = table.sort_values("RMSE", ascending=False)
    colors = [SERIES[0] if m == highlight else MUTED if m in benchmarks else BLUE_RAMP["200"] for m in t["model"]]
    fig = go.Figure(go.Bar(y=t["model"], x=t["RMSE"].round(3), orientation="h", marker_color=colors,
                           text=[f"{r:.2f} pp" for r in t["RMSE"]], textposition="outside",
                           customdata=np.c_[t["rel. RMSE"], t["DM p-value"]],
                           hovertemplate="%{y}<br>RMSE %{x:.3f} pp<br>vs no change: %{customdata[0]:.2f}"
                                         "<br>DM p-value %{customdata[1]:.3f}<extra></extra>"))
    row = table.set_index("model").loc[highlight]
    fig.update_layout(height=60 * len(t) + 140, margin=dict(l=230, b=50), xaxis_range=[0, 1.18 * t["RMSE"].max()],
                      title=_title(f"{highlight}: RMSE {row['RMSE']:.2f} pp, "
                                   f"{100 * (1 - row['rel. RMSE']):.0f}% below the no-change forecast",
                                   f"3-month-ahead headline YoY, out-of-sample, {int(row['n'])} forecasts; "
                                   f"Diebold-Mariano p = {row['DM p-value']:.2f}"))
    return fig


def track_record_chart(bt: pd.DataFrame, headline: pd.Series, models: list[str], h: int = 3,
                       since: str = "2019-01") -> go.Figure:
    actual = yoy(headline).dropna().loc[since:]
    fig = go.Figure()
    fig.add_scatter(x=_ts(actual.index), y=actual.round(2), name="Published", line=dict(color=INK, width=2))
    for i, m in enumerate(models):
        d = bt[(bt["model"] == m) & (bt["h"] == h)].set_index("target")["forecast"].loc[since:]
        fig.add_scatter(x=_ts(d.index), y=d.round(2), name=m,
                        line=dict(color=SERIES[i], width=2))
    fig.update_layout(hovermode="x unified", yaxis_ticksuffix="%",
                      title=_title("Track record: forecasts made three months ahead vs the outcome",
                                   "Euro area HICP annual rate, %"))
    return _fit_legend(fig)


# ---- pass-through and shocks -----------------------------------------------------

def passthrough_chart(lp: dict[str, pd.DataFrame], labels: dict[str, str], components: list[str],
                      energy: str = "NRG", sample: tuple[str, str] | None = None) -> go.Figure:
    comps = [c for c in components if c != energy and c in lp]
    colors = component_colors(components)
    fig = make_subplots(rows=1, cols=len(comps), subplot_titles=[labels.get(c, c) for c in comps],
                        shared_yaxes=True)
    for j, c in enumerate(comps, start=1):
        color = colors[c]
        d = lp[c]
        fig.add_scatter(x=[*d.index, *d.index[::-1]], y=[*d["hi"], *d["lo"][::-1]], fill="toself",
                        fillcolor="rgba(137,135,129,0.18)", line=dict(width=0), hoverinfo="skip",
                        showlegend=False, row=1, col=j)
        fig.add_scatter(x=d.index, y=d["beta"].round(3), line=dict(color=color, width=2), name=labels.get(c, c),
                        showlegend=False, row=1, col=j,
                        hovertemplate="h=%{x} months: %{y:.3f}<extra></extra>")
        fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=j)
    fig.update_xaxes(title_text="months after the energy price move")
    fig.update_layout(height=420, margin=dict(b=70), title=_title(
        "Energy price moves pass through to other prices with a lag",
        "Cumulative % response to a 1% rise in the energy HICP (local projections"
        + (f", origins and outcomes in {sample[0][:4]}–{sample[1][:4]}" if sample else "") + ", 90% HAC bands)"))
    return fig


def shock_chart(result, labels: dict[str, str], title: str) -> go.Figure:
    y, d, comp = result.yoy, result.delta, result.by_component
    x = _ts(y.index)
    # show components carrying at least 2% of the peak impact; the rest (incl. chain-linking
    # interaction terms) is grouped, so a shock on one division shows one colour
    threshold = max(1e-3, 0.02 * d["total"].abs().max())
    big = comp.abs().max()
    keep = list(big[big >= threshold].sort_values(ascending=False).index[: len(SERIES) - 1])
    fixed = component_colors(list(comp.columns)) if comp.shape[1] <= len(SERIES) else None
    comp = keep_and_other(comp, keep)
    colors = fixed or component_colors(list(comp.columns))
    colors[OTHER] = MUTED
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
    fig.add_scatter(x=x, y=y["baseline"].round(2), name="Baseline: central forecast, then conditional path",
                    line=dict(color=MUTED, width=2), row=1, col=1)
    fig.add_scatter(x=x, y=y["shocked"].round(2), name="With shock", line=dict(color=INK, width=2), row=1, col=1)
    for c in comp.columns:
        if comp[c].abs().max() < threshold:
            continue
        vals = comp[c].round(3).where(comp[c].abs() > 1e-4)  # no zero-height stubs before the shock
        fig.add_bar(x=x, y=vals, name=labels.get(c, c), row=2, col=1,
                    marker=dict(color=colors[c], line=dict(color=SURFACE, width=1)))
    peak = d["total"].abs().idxmax()
    split = ("Direct effects only (accounting, no estimated coefficient)" if d["indirect"].abs().max() < 1e-9 else
             f"Direct {d['direct'].loc[peak]:+.2f} pp, indirect (exploratory) {d['indirect'].loc[peak]:+.2f} pp at the peak")
    fig.update_layout(barmode="relative", hovermode="x unified", height=640,
                      title=_title(f"{title}: peak impact {d['total'].loc[peak]:+.2f} pp in {peak.strftime('%b %Y')}", split))
    if result.forecast_end is not None:  # baseline = central forecast up to here, a conditional path after
        a, b = result.forecast_end.to_timestamp(), (result.forecast_end + 1).to_timestamp()
        edge = a + (b - a) / 2  # between the last forecast month and the next
        fig.add_vline(x=edge, line=dict(color=INK_2, width=1, dash="dot"), row="all", col=1)
        for text, anchor in (("central forecast  ", "right"), ("  conditional path", "left")):
            fig.add_annotation(x=edge, y=1, xref="x", yref="y domain", text=text, xanchor=anchor, yanchor="bottom",
                               showarrow=False, font=dict(size=11, color=INK_2))
    fig.update_yaxes(ticksuffix="%", title_text="Headline YoY", row=1, col=1)
    fig.update_yaxes(title_text="Impact, pp", row=2, col=1)
    return _fit_legend(fig)


def shock_country_chart(result, top: int = 12) -> go.Figure:
    peak = result.delta["total"].abs().idxmax()
    s = result.by_country.loc[peak].sort_values(key=np.abs, ascending=False).iloc[:top][::-1]
    fig = go.Figure(go.Bar(y=list(s.index), x=s.round(3), orientation="h", marker_color=SERIES[0],
                           hovertemplate="%{y}: %{x:.3f} pp<extra></extra>"))
    fig.update_layout(height=420, margin=dict(b=50), title=_title(f"Where the impact comes from ({peak.strftime('%b %Y')})",
                                               "Contribution of each country to the change in euro area headline YoY, pp"))
    return fig
