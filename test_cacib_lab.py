"""Tests de comportement : fuite temporelle, panier, chaînage, scénarios et gel."""

from dataclasses import replace
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cacib_lab.data import (BRANCHES, DIVISIONS, Protocol, Snapshot, digest,
    month, tail_history, verify_inputs, write_json, read_json)
from cacib_lab.evaluation import ForecastEngine, metrics, validate_grid, origin_result
from cacib_lab.hierarchy import aggregate, reconcile, weights_at, leaf_paths
from cacib_lab.models import predict, ridge_external
from cacib_lab.scenarios import Shock, apply_shocks, decomposition, simulate
from cacib_lab.uncertainty import (interval_score, joint_headline_draws,
    joint_leaf_draws, mature_errors, paired_block_bootstrap, prequential_intervals)
from cacib_lab.workflow import unpack_scenario


def synthetic_snapshot():
    index = pd.period_range("2000-01", "2026-09", freq="M")
    base = 70*np.exp(.001*np.arange(len(index)))
    seasonal = np.exp(.003*np.sin(2*np.pi*np.arange(len(index))/12))
    columns = [("EA","TOTAL"), *[("EA",c) for c in BRANCHES],
               *[(g,c) for g in ["FR","DE"] for c in ["TOTAL",*DIVISIONS]]]
    columns = pd.MultiIndex.from_tuples(columns,names=["geo","component"])
    levels = pd.DataFrame(np.repeat((base*seasonal)[:,None],len(columns),axis=1),index=index,columns=columns)
    years = list(range(2000,2027))
    iw = pd.DataFrame(0.,index=years,columns=columns)
    for c in BRANCHES:
        iw[("EA",c)] = 200.
    for g in ["FR","DE"]:
        iw[(g,"TOTAL")] = 1000.
        for c in DIVISIONS:
            iw[(g,c)] = 1000/13
    cw = pd.DataFrame({"FR":500.,"DE":500.},index=years)
    rng=np.random.default_rng(2)
    ext = pd.DataFrame(rng.normal(size=(len(index),6)),index=index,columns=[f"p{i}" for i in range(6)])
    return Snapshot(levels,iw,cw,ext,Protocol(),{"membership":{"FR":"1999-01","DE":"1999-01"}}, {"FR":"France","DE":"Allemagne"})


def scenario_fixture(origin="2026-09"):
    s=synthetic_snapshot()
    top,paths,history,w,notes=leaf_paths(s,month(origin),"countries","seasonal_mean")
    return s,history,paths,w,s.headline.loc[:origin]


def calibration_records(model="direct__ets_additive",n=60):
    rows=[]
    for i,origin in enumerate(pd.period_range("2014-01",periods=n,freq="M")):
        for h in [1,2,3]:
            pred=100.+h
            actual=pred*np.exp(.005*np.sin(i)+.001*h)
            rows.append({"origin":str(origin),"target":str(origin+h),"horizon":h,
                         "model":model,"predicted_index":pred,"actual_index":actual})
    return pd.DataFrame(rows)


class DataAndModelsTests(unittest.TestCase):
    def test_protocol_rejects_test_overlap(self):
        with self.assertRaises(ValueError):
            Protocol(test_start="2024-09")

    def test_tail_does_not_interpolate_gap(self):
        x=pd.Series([1.,2.,np.nan,4.,5.],index=pd.period_range("2020-01",periods=5,freq="M"))
        result=tail_history(x,"2020-05")
        self.assertEqual(result.tolist(),[4.,5.])

    def test_forecast_invariant_to_unseen_future(self):
        s=synthetic_snapshot()
        origin=month("2018-12")
        changed=copy.deepcopy(s)
        changed.levels.loc[origin+1:]*=4
        changed.external.loc[origin+1:]*=1000
        for model in ["direct__ridge","direct__ridge_ppi","countries__seasonal_mean"]:
            a=ForecastEngine(s,origin).forecast(model)
            b=ForecastEngine(changed,origin).forecast(model)
            np.testing.assert_allclose(a,b,atol=1e-11)

    def test_external_folds_respect_label_availability(self):
        s=synthetic_snapshot()
        forecast,info=ridge_external(s.headline.loc[:"2018-12"],s.external)
        for horizon,data in info.items():
            self.assertLessEqual(month(data["last_label"]),month("2018-12"))
            for fold in data["folds"]:
                self.assertLessEqual(month(fold["last_train_label"]),month(fold["first_validation_origin"]))
        self.assertTrue(forecast.gt(0).all())

    def test_recent_gap_uses_logged_past_value(self):
        s=synthetic_snapshot()
        x=s.headline.copy()
        x.loc["2026-09"]=np.nan
        result,info=predict(x,"2026-09","seasonal_mean",allow_fallback=True)
        self.assertTrue(info["fallback"])
        self.assertEqual(info["last_observation"],"2026-08")
        self.assertTrue(result.eq(x.loc["2026-08"]).all())

    def test_direct_failure_is_not_silently_replaced(self):
        x=pd.Series([1.,2.,3.],index=pd.period_range("2020-01",periods=3,freq="M"))
        with self.assertRaises(ValueError):
            predict(x,"2020-03","ets_additive")

    def test_input_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            file=root/"data.bin"; file.write_bytes(b"original")
            lock={"sha256":{"data.bin":digest(file)}}
            with patch("cacib_lab.data.project_root",return_value=root):
                verify_inputs(lock)
                file.write_bytes(b"changed")
                with self.assertRaises(ValueError): verify_inputs(lock)

    def test_atomic_json_refuses_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): write_json(Path(temp)/"x.json",{"x":float("nan")})


class AggregationTests(unittest.TestCase):
    def test_complete_country_basket_sums_to_one(self):
        s=synthetic_snapshot()
        w=weights_at(s,2024,"countries")
        self.assertAlmostEqual(w.sum(),1.)
        self.assertEqual(len(w),26)

    def test_membership_is_date_specific(self):
        s=synthetic_snapshot()
        s.identity["membership"]["DE"]="2023-01"
        s.country_weights.loc[:2022,"FR"]=1000.
        s.country_weights.loc[:2022,"DE"]=np.nan
        self.assertEqual(set(weights_at(s,2022,"countries").index.get_level_values(0)),{"FR"})
        self.assertEqual(set(weights_at(s,2023,"countries").index.get_level_values(0)),{"FR","DE"})

    def test_incomplete_positive_basket_raises(self):
        s=synthetic_snapshot()
        s.item_weights.loc[2024,("FR","CP01")]=np.nan
        with self.assertRaises(ValueError): weights_at(s,2024,"countries")

    def test_zero_weight_missing_series_is_harmless(self):
        s,h,p,w,top=scenario_fixture()
        key=p.columns[0]
        w=w.copy();w[key]=0;w/=w.sum()
        p[key]=np.nan
        result=aggregate(h,p,w,top)
        self.assertTrue(np.isfinite(result).all())

    def test_positive_weight_missing_series_raises(self):
        s,h,p,w,top=scenario_fixture()
        p.iloc[0,0]=np.nan
        with self.assertRaises(ValueError): aggregate(h,p,w,top)

    def test_reconciliation_matches_every_month_across_january(self):
        s,h,p,w,top=scenario_fixture("2025-11")
        target=aggregate(h,p,w,top)*pd.Series([1.01,.99,1.02],index=p.index)
        adjusted,factors=reconcile(h,p,w,top,target)
        np.testing.assert_allclose(aggregate(h,adjusted,w,top),target,rtol=1e-11)
        self.assertTrue(adjusted.gt(0).all().all())

    def test_no_future_weight_used(self):
        s=synthetic_snapshot()
        a=leaf_paths(s,"2025-11","countries","seasonal_mean")[0]
        s.item_weights.loc[2026]*=50
        s.country_weights.loc[2026]*=8
        b=leaf_paths(s,"2025-11","countries","seasonal_mean")[0]
        np.testing.assert_allclose(a,b)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.s,self.h,self.p,self.w,self.top=scenario_fixture("2025-11")

    def shock(self,magnitude=10,kind="permanent",duration=1):
        return Shock(("FR",),("CP07",),magnitude,str(self.p.index[0]),kind,duration)

    def test_zero_shock_preserves_baseline(self):
        r=simulate(self.h,self.p,self.w,self.top,[self.shock(0)])
        np.testing.assert_allclose(r.impact_pp,0,atol=1e-12)

    def test_permanent_shock_is_price_level_step(self):
        changed=apply_shocks(self.p,[self.shock()])
        np.testing.assert_allclose(changed[("FR","CP07")]/self.p[("FR","CP07")],1.1)

    def test_temporary_shock_reverts_to_baseline(self):
        changed=apply_shocks(self.p,[self.shock(kind="temporary",duration=1)])
        np.testing.assert_allclose(changed[("FR","CP07")]/self.p[("FR","CP07")],[1.1,1,1])
        result=simulate(self.h,self.p,self.w,self.top,[self.shock(kind="temporary",duration=1)])
        self.assertGreater(result.impact_pp.iloc[0],0)
        # Au chaînage de janvier, la base de décembre et l'agrégat changent.
        # Avec des poids maintenus fixes, un résidu de chaînage est possible.
        share=self.w[("FR","CP07")]
        baseline=aggregate(self.h,self.p,self.w,self.top)
        expected_factor=(1+.1*share)*(1-share+share/1.1)
        for i in [1,2]:
            expected=100*baseline.iloc[i]/self.top.loc[self.p.index[i]-12]*(expected_factor-1)
            self.assertAlmostEqual(result.impact_pp.iloc[i],expected,places=10)

    def test_temporary_shock_without_year_change_has_zero_later_impact(self):
        s,h,p,w,top=scenario_fixture("2026-06")
        shock=Shock(("FR",),("CP07",),10,str(p.index[0]),"temporary",1)
        result=simulate(h,p,w,top,[shock])
        np.testing.assert_allclose(result.impact_pp.iloc[1:],0,atol=1e-10)

    def test_monthly_shock_compounds_then_holds_level(self):
        changed=apply_shocks(self.p,[self.shock(kind="monthly",duration=2)])
        np.testing.assert_allclose(changed[("FR","CP07")]/self.p[("FR","CP07")],[1.1,1.21,1.21])

    def test_pp_impact_differs_from_percent_price_shock(self):
        result=simulate(self.h,self.p,self.w,self.top,[self.shock()])
        expected=100*(aggregate(self.h,self.p,self.w,self.top).iloc[0]/self.top.loc[self.p.index[0]-12])*.1*self.w[("FR","CP07")]
        self.assertAlmostEqual(result.impact_pp.iloc[0],expected,places=10)
        self.assertLess(result.impact_pp.iloc[0],1)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError): apply_shocks(self.p,[self.shock(),self.shock()])

    def test_nonpositive_price_shock_rejected(self):
        with self.assertRaises(ValueError): self.shock(-100)

    def test_fractional_duration_rejected(self):
        with self.assertRaises(ValueError): self.shock(duration=1.5)

    def test_start_outside_horizon_rejected(self):
        shock=Shock(("FR",),("CP07",),10,"2020-01")
        with self.assertRaises(ValueError): apply_shocks(self.p,[shock])

    def test_sequential_contributions_sum_exactly(self):
        shocks=[self.shock(),Shock(("DE",),("CP01",),5,str(self.p.index[0]))]
        result=simulate(self.h,self.p,self.w,self.top,shocks)
        contributions=decomposition(self.h,self.p,self.w,self.top,shocks).groupby("target").impact_pp.sum()
        np.testing.assert_allclose(contributions.to_numpy(),result.impact_pp,atol=1e-10)


class EvaluationTests(unittest.TestCase):
    def test_failed_forecast_disqualifies_metric(self):
        frame=pd.DataFrame({"model":["x","x"],"horizon":[3,3],"status":["ok","failed"],"error_pp":[1,np.nan],"fallback_count":[0,0]})
        score=metrics(frame).iloc[0]
        self.assertFalse(score.eligible)
        self.assertEqual(score.failures,1)
        self.assertTrue(pd.isna(score.rmse_pp))

    def test_missing_date_is_not_silently_removed(self):
        frame=pd.DataFrame([{"model":"x","horizon":h,"target":"2020-01"} for h in [1,2]])
        with self.assertRaises(ValueError): validate_grid(frame,["x"],"2020-01","2020-01")

    def test_validation_output_never_contains_test_target(self):
        s=synthetic_snapshot()
        r=origin_result(s,month("2024-08"),["direct__flat_yoy"],month("2014-01"),month("2024-09"))
        self.assertTrue(all(x["target"]<"2024-10" for x in r["records"]+r["leaves"]))
        self.assertEqual(len(r["records"]),1)

    def test_future_targets_excluded_from_calibration(self):
        records=calibration_records()
        a=mature_errors(records,"2016-01","direct__ets_additive")
        records.loc[records.target.gt("2016-01"),"actual_index"]*=10
        b=mature_errors(records,"2016-01","direct__ets_additive")
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(all(month(o)+3<=month("2016-01") for o in a.index))

    def test_joint_draws_preserve_cross_horizon_errors(self):
        frame=calibration_records()
        point=pd.Series([101,102,103],index=pd.period_range("2020-01",periods=3,freq="M"))
        draws,info=joint_headline_draws(point,frame,"direct__ets_additive","2019-12",draws=100)
        logerrors=np.log(draws/point.to_numpy())/np.array(info["volatility_scale"])
        np.testing.assert_allclose(logerrors[:,1]-logerrors[:,0],.001,atol=1e-12)

    def test_partial_horizon_uses_correct_residual_column(self):
        frame=calibration_records()
        point=pd.Series([103.],index=pd.PeriodIndex(["2020-03"],freq="M"))
        draws,_=joint_headline_draws(point,frame,"direct__ets_additive","2019-12",draws=50)
        self.assertEqual(draws.shape,(50,1))

    def test_joint_leaf_draws_preserve_cross_component_dependence(self):
        s,h,p,w,top=scenario_fixture("2025-11")
        rows=[]
        for i,o in enumerate(pd.period_range("2014-01",periods=40,freq="M")):
            for horizon in [1,2,3]:
                error=.005*np.sin(i)+.001*horizon
                for geo,c in p.columns:
                    rows.append({"model":"countries__seasonal_mean","origin":str(o),
                        "target":str(o+horizon),"horizon":horizon,"geo":geo,"component":c,
                        "predicted_index":100.,"actual_index":100*np.exp(error)})
        draws,info=joint_leaf_draws(s,"2025-11",p,"countries__seasonal_mean",pd.DataFrame(rows),draws=50)
        baseline=aggregate(h,p,w,top)
        errors=np.log(draws/baseline.to_numpy())
        np.testing.assert_allclose(errors[:,1]-errors[:,0],.001,atol=1e-12)
        self.assertIn("chain_aggregation",info["method"])

    def test_insufficient_calibration_is_explicit(self):
        frame=calibration_records(n=10)
        point=pd.Series([101,102,103],index=pd.period_range("2020-01",periods=3,freq="M"))
        draws,info=joint_headline_draws(point,frame,"direct__ets_additive","2019-12")
        self.assertIsNone(draws)
        self.assertEqual(info["calibration_origins"],10)

    def test_interval_score_penalizes_noncoverage(self):
        self.assertEqual(interval_score(2,1,3,.8),2)
        self.assertGreater(interval_score(5,1,3,.8),2)

    def test_block_bootstrap_identifies_identical_losses(self):
        rows=[]
        for i,t in enumerate(pd.period_range("2020-01",periods=24,freq="M")):
            for m in ["a","b"]:
                rows.append({"target":str(t),"model":m,"horizon":3,"status":"ok","error_pp":np.sin(i)})
        result=paired_block_bootstrap(pd.DataFrame(rows),benchmark="a",draws=50)
        self.assertTrue(result.ci95_low.eq(0).all() and result.ci95_high.eq(0).all())


class FreezeTests(unittest.TestCase):
    def test_numerical_change_blocks_final_test(self):
        from cacib_lab.workflow import verify_selection
        s=synthetic_snapshot()
        with patch("cacib_lab.workflow.read_json",return_value={"inputs":"different","numerics":{}}):
            with self.assertRaisesRegex(ValueError,"diffèrent"):
                verify_selection(s)


def dashboard_fixture():
    s,h,p,w,top=scenario_fixture()
    keys=list(w.index)
    point=aggregate(h,p,w,top)
    forecast=[]
    for i,t in enumerate(p.index,1):
        y=float(100*(point[t]/top[t-12]-1))
        forecast.append({"target":str(t),"horizon":i,"predicted_index":float(point[t]),
                         "predicted_yoy":y,"lower_80":y-.2,"upper_80":y+.2,
                         "lower_95":y-.4,"upper_95":y+.4})
    rows=[{"model":"direct__ets_additive","horizon":horizon,"eligible":True,
           "mae_pp":.4,"rmse_pp":.5,"bias_pp":.1,"failures":0,"origins_with_fallback":0} for horizon in [1,2,3]]
    return {"schema_version":1,"metadata":{"as_of":"2026-09","selected_model":"direct__ets_additive",
        "protocol":{"validation_start":"2014-01","validation_end":"2024-09","test_start":"2024-10","test_end":"2026-09"},
        "calibration_origins":40,"interval_method":"joint_headline_log_errors","limitations":["Test synthétique"],
        "collection_id":"synthetic-test","generated_at":"2026-10-27"},
        "history":[{"month":str(t),"index":float(top[t]),"yoy":float(100*(top[t]/top[t-12]-1))} for t in top.index[-24:]],
        "forecast":forecast,"validation_metrics":rows,"test_metrics":rows,"interval_metrics":[],
        "countries":{"FR":"France","DE":"Allemagne"},"components":{"CP07":"Transport"},
        "scenario":{"columns":[list(k) for k in keys],"weights":w.tolist(),
            "future_months":p.index.astype(str).tolist(),"future_values":p.to_numpy().tolist(),
            "history_months":h.index[-13:].astype(str).tolist(),"history_values":h.iloc[-13:].to_numpy().tolist(),
            "headline_months":top.index[-13:].astype(str).tolist(),"headline_values":top.iloc[-13:].tolist()}}


@unittest.skipUnless(importlib.util.find_spec("dash") and importlib.util.find_spec("dash_mantine_components"),"Installer requirements-lab.txt pour les tests du dashboard.")
class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cacib_lab.app import create_app
        cls.app=create_app(dashboard_fixture())
        cls.client=cls.app.server.test_client()

    def test_layout_and_health_routes(self):
        self.assertEqual(self.client.get("/").status_code,200)
        self.assertEqual(self.client.get("/_dash-layout").status_code,200)
        self.assertTrue(self.client.get("/healthz").get_json()["ready"])

    def test_scenario_callback_changes_forecast(self):
        key=next(k for k in self.app.callback_map if k.startswith("..scenario-chart.figure"))
        outputs=[{"id":i,"property":p} for i,p in [
            ("scenario-chart","figure"),("impact-kpi","children"),("scenario-table","children"),
            ("attribution-chart","figure"),("scenario-message","children"),("scenario-store","data")]]
        states=[("shock-countries",["FR"]),("shock-components",["CP07"]),("shock-magnitude",10),
                ("shock-kind","permanent"),("shock-start","2026-10"),("shock-duration",1)]
        response=self.client.post("/_dash-update-component",json={"output":key,"outputs":outputs,
            "inputs":[{"id":"simulate","property":"n_clicks","value":1}],
            "state":[{"id":i,"property":"value","value":v} for i,v in states],"changedPropIds":["simulate.n_clicks"]})
        self.assertEqual(response.status_code,200)
        data=response.get_json()["response"]["scenario-store"]["data"]
        self.assertGreater(data[-1]["impact_pp"],0)

    def test_forecast_csv_export(self):
        response=self.client.post("/_dash-update-component",json={"output":"forecast-download.data",
            "outputs":{"id":"forecast-download","property":"data"},
            "inputs":[{"id":"export-forecast","property":"n_clicks","value":1}],"state":[],
            "changedPropIds":["export-forecast.n_clicks"]})
        self.assertEqual(response.status_code,200)
        self.assertIn("predicted_yoy",response.get_json()["response"]["forecast-download"]["data"]["content"])

    def test_bundle_roundtrip_preserves_zero_shock(self):
        bundle=dashboard_fixture()
        data=unpack_scenario(bundle)
        result=simulate(*data,[])
        np.testing.assert_allclose(result.impact_pp,0,atol=1e-12)

    def test_missing_bundle_reports_not_ready(self):
        from cacib_lab.app import create_app
        with patch.dict("os.environ",{"CACIB_BUNDLE_PATH":"/this/file/does/not/exist.json"}):
            app=create_app()
        self.assertEqual(app.server.test_client().get("/healthz").status_code,503)


if __name__ == "__main__":
    unittest.main()
