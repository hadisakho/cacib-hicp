# Export pour l'application

La commande `uv run python finish_project.py forecast` crée ici
`dashboard.json` après le test final. Il contient les prévisions, les métriques
et les trajectoires nécessaires aux scénarios. Il ne contient ni identifiants,
ni données privées : ses observations économiques proviennent d'Eurostat.

Ce fichier doit être présent dans le dépôt utilisé pour héberger l'application.
Les jeux bruts `data/` et les caches `reports/` ne sont pas requis par le serveur.

Si `deploy/` est ignoré par le `.gitignore` existant :

```bash
git add -f deploy/dashboard.json
```

Les résultats fournis dans `example_results/` proviennent de l'extraction de
vérification. Ils sont séparés des résultats de ton exécution.
