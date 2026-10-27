"""Dashboard Dash + Mantine. Lecture d'un export, calcul de chocs en mémoire.

Lancement local : python finish_project.py serve
Production : gunicorn cacib_lab.app:server --bind 0.0.0.0:$PORT
"""

import os
from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import read_json
from .evaluation import MODEL_LABELS
from .scenarios import Shock, decomposition, simulate
from .workflow import unpack_scenario

NAVY, TEAL, GOLD = "#142e40", "#087f75", "#b67d20"


def chart_style(figure, title=None, ytitle="Inflation annuelle (%)"):
    figure.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color=NAVY, size=13),
        title=dict(text=title or "", font=dict(size=16)), margin=dict(l=48,r=24,t=42,b=42),
        height=420, hovermode="x unified", legend=dict(orientation="h", y=-.16),
        xaxis=dict(showgrid=False), yaxis=dict(title=ytitle, gridcolor="#edf0f2", zerolinecolor="#d8e0e6"))
    return figure


def forecast_figure(bundle):
    hist = pd.DataFrame(bundle["history"])
    forecast = pd.DataFrame(bundle["forecast"])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=hist.month, y=hist.yoy, name="Observé", line=dict(color=NAVY,width=2.5)))
    last_x, last_y = hist.month.iloc[-1], hist.yoy.iloc[-1]
    x = [last_x, *forecast.target.tolist()]
    for level, color in [(95,"rgba(8,127,117,.08)"),(80,"rgba(8,127,117,.17)")]:
        figure.add_trace(go.Scatter(x=x,y=[last_y,*forecast[f"upper_{level}"]],
                                    mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=x,y=[last_y,*forecast[f"lower_{level}"]],
                        fill="tonexty",fillcolor=color,mode="lines",line=dict(width=0),name=f"Intervalle {level} %"))
    figure.add_trace(go.Scatter(x=x,y=[last_y,*forecast.predicted_yoy],name="Prévision centrale",
                    mode="lines+markers",line=dict(color=TEAL,width=3,dash="dot")))
    return chart_style(figure, "Inflation de la zone euro · historique et prévision")


def scenario_result(bundle, countries, components, magnitude, kind, start, duration):
    """Fonction pure partagée par callbacks et tests HTTP."""
    if magnitude is None or duration is None:
        raise ValueError("Renseigner l'amplitude et la durée.")
    shock = Shock(tuple(countries or []), tuple(components or []), float(magnitude),
                  start, kind, int(duration))
    data = unpack_scenario(bundle)
    result = simulate(*data, [shock])
    attribution = decomposition(*data, [shock])
    return result, attribution


def scenario_figure(result):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=result.target,y=result.baseline_yoy,name="Trajectoire centrale",
                    mode="lines+markers",line=dict(color=NAVY,width=3)))
    figure.add_trace(go.Scatter(x=result.target,y=result.scenario_yoy,name="Avec choc",
                    mode="lines+markers",line=dict(color=GOLD,width=3)))
    return chart_style(figure,"Effet conditionnel du choc sur l'inflation")


def comparison_figure(bundle, phase, horizon):
    data = pd.DataFrame(bundle[f"{phase}_metrics"])
    figure = go.Figure()
    if len(data):
        data = data.loc[data.horizon.eq(horizon) & data.eligible].sort_values("rmse_pp",ascending=False)
        figure.add_trace(go.Bar(x=data.rmse_pp,y=data.model.map(lambda m:MODEL_LABELS.get(m,m)),
                               orientation="h",marker_color=[TEAL if m==bundle['metadata']['selected_model'] else "#b7c7d1" for m in data.model],
                               hovertemplate="%{x:.3f} point de pourcentage<extra></extra>"))
    figure = chart_style(figure, "RMSE par modèle", "")
    figure.update_layout(xaxis_title="RMSE (points de pourcentage)",margin=dict(l=255,r=25,t=45,b=55))
    return figure


def table(data, columns=None):
    frame = pd.DataFrame(data)
    if columns:
        frame = frame.reindex(columns=columns)
    for c in frame.select_dtypes(include="number"):
        frame[c] = frame[c].round(3)
    if "model" in frame:
        frame["model"] = frame["model"].map(lambda m:MODEL_LABELS.get(m,m))
    return dash_table.DataTable(data=frame.to_dict("records"),
        columns=[{"name": c.replace("_"," "),"id": c} for c in frame.columns],
        sort_action="native",page_size=12,
        style_table={"overflowX":"auto"}, style_header={"backgroundColor":"#f1f5f6","fontWeight":600,"border":"none"},
        style_cell={"fontFamily":"Arial, sans-serif","fontSize":13,"padding":"12px","textAlign":"left","border":"none","borderBottom":"1px solid #edf0f2"})


def paper(children, **kwargs):
    return dmc.Paper(children,p="xl",radius="md",withBorder=True,**kwargs)


def create_app(bundle=None):
    app = Dash(__name__,title="Euro Inflation Lab",assets_folder=str(Path(__file__).parent/"assets"))
    if bundle is None:
        path = Path(os.environ.get("CACIB_BUNDLE_PATH", "deploy/dashboard.json"))
        if not path.exists():
            app.layout = dmc.MantineProvider(dmc.Container([
                dmc.Title("Euro Inflation Lab",order=1),
                dmc.Alert("Exécuter la chaîne de validation, le test final puis la commande forecast pour produire deploy/dashboard.json.",title="Prévision à préparer",color="teal")],py=60))
            @app.server.get("/healthz")
            def missing_health():
                return {"ready": False,"reason":"forecast_bundle_missing"},503
            return app
        bundle = read_json(path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Version d'export du dashboard non reconnue.")
    meta = bundle["metadata"]
    forecast = pd.DataFrame(bundle["forecast"])
    history = pd.DataFrame(bundle["history"])
    countries = [{"value":g,"label":label} for g,label in bundle["countries"].items()]
    components = [{"value":c,"label":f"{c} · {label}"} for c,label in bundle["components"].items()]
    future = bundle["scenario"]["future_months"]
    default_country = "FR" if "FR" in bundle["countries"] else countries[0]["value"]
    default_component = "CP07"
    last = forecast.iloc[-1]
    kpis = [
        paper([dmc.Text("DERNIÈRE OBSERVATION",size="xs",c="dimmed",fw=700,lts=1),
               dmc.Text(f"{history.yoy.iloc[-1]:.2f} %",className="kpi-number"),dmc.Text(meta["as_of"],size="sm",c="dimmed")]),
        paper([dmc.Text("PRÉVISION À TROIS MOIS",size="xs",c="dimmed",fw=700,lts=1),
               dmc.Text(f"{last.predicted_yoy:.2f} %",className="kpi-number",c=TEAL),dmc.Text(last.target,size="sm",c="dimmed")]),
        paper([dmc.Text("INTERVALLE À 80 %",size="xs",c="dimmed",fw=700,lts=1),
               dmc.Text(f"{last.lower_80:.2f} à {last.upper_80:.2f} %",className="kpi-number"),
               dmc.Text("Couverture empirique évaluée hors échantillon",size="xs",c="dimmed")])]
    overview = html.Div([
        dmc.SimpleGrid(kpis,cols={"base":1,"sm":3},spacing="lg",mb="lg"),
        paper(dcc.Graph(figure=forecast_figure(bundle),config={"displaylogo":False})),
        dmc.Group([dmc.Text(f"Modèle : {MODEL_LABELS.get(meta['selected_model'],meta['selected_model'])}",size="sm"),
                   dmc.Button("Exporter les prévisions",id="export-forecast",variant="light",color="teal")],justify="space-between",mt="lg"),
        dcc.Download(id="forecast-download"),
        dmc.Text("Les intervalles représentent l'incertitude de prévision; ils ne garantissent pas la réalisation future.",size="xs",c="dimmed",mt="sm")])
    model_view = html.Div([
        dmc.Group([
            dmc.SegmentedControl(id="metric-phase",value="validation",data=[{"label":"Validation","value":"validation"},{"label":"Test final","value":"test"}]),
            dmc.Select(id="metric-horizon",label="Horizon",value="3",data=[{"value":str(h),"label":f"{h} mois"} for h in [1,2,3]],allowDeselect=False,w=150)],justify="space-between",mb="lg"),
        paper(dcc.Graph(id="comparison-chart",figure=comparison_figure(bundle,"validation",3),config={"displaylogo":False})),
        paper(html.Div(id="comparison-table"),mt="lg"),
        paper([dmc.Title("Intervalles · test final",order=4,mb="md"),table(bundle["interval_metrics"])],mt="lg"),
        dmc.Text("RMSE, MAE et biais en points de pourcentage. Les dates d'échec restent dans la grille d'évaluation.",size="xs",c="dimmed",mt="sm")])
    controls = paper([
        dmc.Text("DÉFINIR LE CHOC",size="xs",fw=700,lts=1,c=TEAL,mb="lg"),
        dmc.MultiSelect(id="shock-countries",label="Pays",data=countries,value=[default_country],searchable=True,mb="md"),
        dmc.MultiSelect(id="shock-components",label="Catégories COICOP v2",data=components,value=[default_component],searchable=True,mb="md"),
        dmc.NumberInput(id="shock-magnitude",label="Amplitude sur les prix (%)",value=10,min=-99,max=100,step=.5,decimalScale=2,mb="md"),
        dmc.Select(id="shock-kind",label="Profil",value="permanent",allowDeselect=False,data=[
            {"value":"permanent","label":"Hausse permanente du niveau"},
            {"value":"temporary","label":"Hausse temporaire du niveau"},
            {"value":"monthly","label":"Hausse mensuelle répétée"}],mb="md"),
        dmc.Select(id="shock-start",label="Début",value=future[0],data=future,allowDeselect=False,mb="md"),
        dmc.NumberInput(id="shock-duration",label="Durée (mois)",value=1,min=1,max=3,step=1,allowDecimal=False,mb="lg"),
        dmc.Group([dmc.Button("Simuler",id="simulate",color="teal"),dmc.Button("Réinitialiser",id="reset",variant="subtle",color="gray")]),
        dmc.Text("La durée s'applique aux chocs temporaires et répétés. Après une hausse répétée, le niveau atteint persiste.",size="xs",c="dimmed",mt="md")])
    scenario_view = html.Div([
        html.Div([controls,html.Div([
            dmc.Alert("Choc imposé aux prix IPCH; poids fixes et trajectoires des autres catégories maintenues.",color="teal",mb="md"),
            html.Div(id="scenario-message"),
            paper([dmc.Text(id="impact-kpi",className="impact-number"),dcc.Graph(id="scenario-chart",config={"displaylogo":False})]),
            paper(html.Div(id="scenario-table"),mt="lg"),
            paper([dmc.Title("Attribution à trois mois",order=4),dcc.Graph(id="attribution-chart",config={"displaylogo":False})],mt="lg"),
            dmc.Button("Exporter le scénario",id="export-scenario",variant="light",color="teal",mt="lg"),
            dcc.Download(id="scenario-download"),dcc.Store(id="scenario-store")])],className="scenario-grid"),
        dmc.Text("Le scénario utilise une allocation proportionnelle des prévisions détaillées vers la prévision globale retenue. L'attribution est comptable et séquentielle; elle dépend de l'ordre en présence d'interactions.",size="xs",c="dimmed",mt="md")])
    methodology = paper([
        dmc.Title("Une chaîne traçable, de la donnée à la simulation",order=3,mb="lg"),
        dmc.Text("Cible : inflation annuelle de la zone euro à un, deux et trois mois. L'horizon principal est trois mois.",mb="md"),
        dmc.Text("L'agrégation utilise les variations depuis décembre précédent, les poids annuels des catégories et les poids des pays. Les chocs sont appliqués aux indices avant réagrégation.",mb="md"),
        dmc.Text(f"Validation : {meta['protocol']['validation_start']} à {meta['protocol']['validation_end']}. Test final : {meta['protocol']['test_start']} à {meta['protocol']['test_end']}.",mb="md"),
        dmc.Text(f"Calibration des intervalles : {meta['calibration_origins']} origines; méthode : {meta['interval_method']}.",mb="md"),
        dmc.Title("Hypothèses et limites",order=4,mb="md"),
        html.Ul([html.Li(t) for t in meta["limitations"]]),
        dmc.Text(f"Collecte : {meta['collection_id']} · Calcul : {meta['generated_at']}",size="xs",c="dimmed",mt="lg"),
        html.A("Méthodologie Eurostat",href="https://ec.europa.eu/eurostat/en/web/hicp/methodology/",target="_blank",rel="noopener noreferrer")])
    app.layout = dmc.MantineProvider(theme={"fontFamily":"Arial, sans-serif","primaryColor":"teal","defaultRadius":"md"},children=[
        html.Div(dmc.Container(dmc.Group([
            html.Div([dmc.Text("RESEARCH / MACROECONOMICS",size="xs",c="#b9d7dc",fw=700,lts=2),
                      dmc.Title("Euro Inflation Lab",order=1,c="white",mt=8)]),
            html.Div([dmc.Badge("EUROSTAT · IPCH",variant="light",color="teal",size="lg"),
                      dmc.Text(f"Données arrêtées à {meta['as_of']}",size="sm",c="#c7d6df",mt=10)])],justify="space-between"),size=1300),className="topbar"),
        dmc.Container([
            dmc.Text("Prévoir l'inflation. Explorer la transmission des chocs.",size="lg",c="dimmed",mb="xl"),
            dmc.Tabs(value="overview",color="teal",children=[
                dmc.TabsList([dmc.TabsTab("Vue d'ensemble",value="overview"),dmc.TabsTab("Modèles et validation",value="models"),dmc.TabsTab("Simulateur de chocs",value="scenarios"),dmc.TabsTab("Méthodologie",value="method")],mb="xl"),
                dmc.TabsPanel(overview,value="overview"),dmc.TabsPanel(model_view,value="models"),
                dmc.TabsPanel(scenario_view,value="scenarios"),dmc.TabsPanel(methodology,value="method")]),
            dmc.Text("Projet de recherche · prévisions conditionnelles · source : Eurostat",size="xs",c="dimmed",mt=50,mb=30)],size=1300,py="xl")])

    @app.server.get("/healthz")
    def health():
        return {"ready":True,"as_of":meta["as_of"],"model":meta["selected_model"]}

    @app.callback(Output("comparison-chart","figure"),Output("comparison-table","children"),Input("metric-phase","value"),Input("metric-horizon","value"))
    def update_models(phase,horizon):
        h=int(horizon)
        frame=pd.DataFrame(bundle[f"{phase}_metrics"])
        frame=frame.loc[frame.horizon.eq(h)]
        return comparison_figure(bundle,phase,h),table(frame,["model","rmse_pp","mae_pp","bias_pp","failures","origins_with_fallback"])

    @app.callback(Output("forecast-download","data"),Input("export-forecast","n_clicks"),prevent_initial_call=True)
    def export_forecast(_):
        return dcc.send_data_frame(forecast.to_csv,"forecast.csv",index=False)

    @app.callback(Output("shock-countries","value"),Output("shock-components","value"),Output("shock-magnitude","value"),Output("shock-kind","value"),Output("shock-start","value"),Output("shock-duration","value"),Input("reset","n_clicks"),prevent_initial_call=True)
    def reset(_):
        return [default_country],[default_component],10,"permanent",future[0],1

    @app.callback(Output("scenario-chart","figure"),Output("impact-kpi","children"),Output("scenario-table","children"),Output("attribution-chart","figure"),Output("scenario-message","children"),Output("scenario-store","data"),
        Input("simulate","n_clicks"),State("shock-countries","value"),State("shock-components","value"),State("shock-magnitude","value"),State("shock-kind","value"),State("shock-start","value"),State("shock-duration","value"))
    def update_scenario(_,selected_countries,selected_components,magnitude,kind,start,duration):
        try:
            result,attribution=scenario_result(bundle,selected_countries,selected_components,magnitude,kind,start,duration)
        except (ValueError,TypeError) as exc:
            return no_update,no_update,no_update,no_update,dmc.Alert(str(exc),color="red",mb="md"),None
        attribution=attribution.loc[attribution.target.eq(future[-1])].copy()
        attribution["cell"]=attribution.geo+" / "+attribution.component
        fig=go.Figure(go.Bar(x=attribution.cell,y=attribution.impact_pp,marker_color=GOLD))
        chart_style(fig,ytitle="Impact (points de pourcentage)")
        impact=f"{result.impact_pp.iloc[-1]:+.3f} point de pourcentage à trois mois"
        return scenario_figure(result),impact,table(result),fig,None,result.to_dict("records")

    @app.callback(Output("scenario-download","data"),Input("export-scenario","n_clicks"),State("scenario-store","data"),prevent_initial_call=True)
    def export_scenario(_,data):
        return dcc.send_data_frame(pd.DataFrame(data).to_csv,"scenario.csv",index=False) if data else no_update
    return app


app = create_app()
server = app.server
