"""Commandes explicites : la validation n'ouvre jamais le test réservé."""

import argparse
from pathlib import Path
import sys

from .data import Protocol, initialize, lab_root, load_snapshot, project_root, read_json, utcnow
from .evaluation import MODELS, run_comparison
from .workflow import analyze, audit, build_forecast, final_test, freeze, unpack_scenario


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Figer extraction et dates; aucun modèle n'est entraîné.")
    init.add_argument("--panel", type=Path)
    init.add_argument("--external", type=Path)
    commands.add_parser("audit", help="Qualité des séries et reconstruction jusqu'à septembre 2024.")
    compare = commands.add_parser("compare", help="Comparer les candidats sur la validation.")
    compare.add_argument("--models", nargs="+", choices=MODELS)
    compare.add_argument("--without-ppi", action="store_true")
    compare.add_argument("--window", type=int, choices=[120])
    compare.add_argument("--restart", action="store_true", help="Nouvelle comparaison, anciens résultats conservés.")
    analysis = commands.add_parser("analyze", help="Erreurs, bootstrap et intervalles préquentiels.")
    analysis.add_argument("--run", type=Path)
    analysis.add_argument("--window", type=int, choices=[120])
    selection = commands.add_parser("freeze", help="Figer le choix à partir de la validation uniquement.")
    selection.add_argument("--run", type=Path)
    selection.add_argument("--window", type=int, choices=[120])
    selection.add_argument("--model", choices=MODELS)
    selection.add_argument("--replace", action="store_true")
    commands.add_parser("test", help="Ouvrir et évaluer le test final avec la procédure figée.")
    commands.add_parser("forecast", help="Prévision actuelle et export du dashboard après le test.")
    scenario = commands.add_parser("scenario", help="Simuler un choc depuis le bundle de prévision.")
    scenario.add_argument("--countries", nargs="+", default=["FR"])
    scenario.add_argument("--components", nargs="+", default=["CP07"])
    scenario.add_argument("--magnitude", type=float, default=10.)
    scenario.add_argument("--kind", choices=["permanent", "temporary", "monthly"], default="permanent")
    scenario.add_argument("--start")
    scenario.add_argument("--duration", type=int, default=1)
    serve = commands.add_parser("serve", help="Ouvrir le dashboard local.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8050)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "init":
        lock = initialize(args.panel, args.external)
        print(f"Extraction figée : {lock['collection_id']}; protocole : {lock['protocol']}")
        return
    if args.command == "serve":
        from .app import app
        app.run(host=args.host, port=args.port, debug=False)
        return
    if args.command == "scenario":
        from .scenarios import Shock, simulate, decomposition
        bundle = read_json(project_root() / "deploy/dashboard.json")
        data = unpack_scenario(bundle)
        shock = Shock(tuple(args.countries), tuple(args.components), args.magnitude,
                      args.start or bundle["scenario"]["future_months"][0], args.kind, args.duration)
        result = simulate(*data, [shock])
        destination = lab_root() / "scenarios" / utcnow().replace(":", "").replace("+", "_")
        destination.mkdir(parents=True)
        result.to_csv(destination / "scenario.csv", index=False)
        decomposition(*data, [shock]).to_csv(destination / "decomposition.csv", index=False)
        print(result.to_string(index=False))
        print(destination)
        return
    snapshot = load_snapshot()
    if args.command == "audit":
        audit(snapshot)
    elif args.command == "compare":
        models = args.models or MODELS.copy()
        if args.without_ppi:
            models = [m for m in models if m != "direct__ridge_ppi"]
        run_comparison(snapshot, models=models, window=args.window, restart=args.restart)
    elif args.command == "analyze":
        analyze(snapshot, args.run, args.window)
    elif args.command == "freeze":
        freeze(snapshot, args.run, args.window, args.model, args.replace)
    elif args.command == "test":
        final_test(snapshot)
    elif args.command == "forecast":
        build_forecast(snapshot)


if __name__ == "__main__":
    main()
