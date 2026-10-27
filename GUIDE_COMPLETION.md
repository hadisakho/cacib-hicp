# CA-CIB : terminer le projet, étape par étape

Cette livraison contient le code de toutes les étapes restantes, hors présentation.
Elle s'ajoute au projet `cacib-hicp` déjà construit. Elle réutilise les données
Eurostat et les fonctions de `baselines.py`, `forecast_models.py`, `ridge_model.py`
et `config.py`. Aucune PCA n'est intégrée.

## 1. Ajouter les fichiers au projet

Décompresser `cacib_completion.zip`. Copier le contenu de `cacib_completion/`
dans la racine de `cacib-hicp`, à côté de `config.py`. Fusionner les dossiers
`notebooks` et `deploy` : conserver notamment le notebook 01 déjà présent.

Sur ton Mac, depuis le terminal du projet, si le dossier extrait est dans Downloads :

```bash
ditto ~/Downloads/cacib_completion .
```

Les fichiers principaux ajoutés sont :

| Fichier | Fonction |
|---|---|
| `finish_project.py` | Commandes communes de la suite |
| `cacib_lab/data.py` | Collecte figée, provenance et protocole |
| `cacib_lab/models.py` | Ridge avec PPI, modèles existants et replis |
| `cacib_lab/hierarchy.py` | Pays, catégories, chaînage et réconciliation |
| `cacib_lab/evaluation.py` | Comparaison, grille exhaustive et reprise |
| `cacib_lab/uncertainty.py` | Intervalles, couverture et bootstrap apparié |
| `cacib_lab/workflow.py` | Audit, sélection, test et prévision courante |
| `cacib_lab/scenarios.py` | Chocs de prix et décomposition de leurs effets |
| `cacib_lab/app.py` | Dashboard Dash + Mantine |
| `test_cacib_lab.py` | Tests de la nouvelle suite |
| `notebooks/02_model_comparison.ipynb` | Synthèse des résultats de validation |
| `notebooks/03_scenarios.ipynb` | Exploration des scénarios |
| `METHODOLOGY.md` | Hypothèses et limites mathématiques |
| `Dockerfile`, `render.yaml` | Préparation de l'hébergement Python |

Les modules sont commentés; le code de calcul reste dans les fichiers Python.
Les notebooks importent leurs fonctions.

## 2. Installer les dépendances et vérifier les tests

Dans `pyproject.toml`, la section `[project]` doit conserver :

```toml
requires-python = ">=3.12"
```

Installer les versions utilisées pour vérifier cette livraison :

```bash
uv add -r requirements-lab.txt
uv add --dev jupyterlab ipykernel nbformat nbclient
uv run python -m unittest test_cacib_lab
```

Lancer ensuite aussi les tests précédents du projet :

```bash
uv run python -m unittest discover -p "test_*.py"
```

Les nouveaux tests couvrent notamment les données futures modifiées, les
labels de Ridge, les poids futurs, les lacunes, le chaînage de janvier, la
réconciliation, les scénarios, la calibration et les callbacks du dashboard.

## 3. Figer l'extraction et contrôler le panel détaillé

Les étapes de téléchargement précédentes restent valables. Si les données et
les PPI ont déjà été préparés, ne pas les retélécharger :

```bash
uv run python finish_project.py init
uv run python finish_project.py audit
```

`init` sélectionne la dernière préparation complète et auditée. Il enregistre
dans `reports/lab/inputs.json` les identifiants, chemins relatifs et SHA-256 des
entrées. Les dates sont explicites : validation janvier 2014–septembre 2024,
test octobre 2024–septembre 2026, coupure septembre 2026. Une seconde commande
`init` identique vérifie le même verrou.

Pour choisir une préparation précise avant le premier verrou :

```bash
uv run python finish_project.py init --panel data/processed/panel/IDENTIFIANT --external data/processed/external/IDENTIFIANT
```

Remplacer les deux `IDENTIFIANT` par les dossiers réellement présents. Cette
commande explicite est une alternative à `init`, pas une étape supplémentaire.

Si les PPI ne sont pas encore préparés, utiliser le script déjà fourni avant
`init` :

```bash
uv run python prepare_external.py
```

L'audit produit :

- `reports/lab/audit/series_quality.csv` : couverture, lacunes, zéros et segment utilisable;
- `historical_reconstruction.csv` : reconstruction stricte par architecture;
- `summary.json` : disponibilité et écarts avec l'indice officiel.

L'analyse économique s'arrête à septembre 2024. Le contrôle de présence des
données peut couvrir toute l'extraction sans utiliser les valeurs du test pour
choisir un modèle. Les règles des séries incomplètes sont dans METHODOLOGY.md.

**Passage :** audit des entrées réussi et écarts de reconstruction examinés.

## 4. Lancer une seule comparaison centrale

```bash
uv run python finish_project.py compare
```

Les huit variantes déclarées avant le test sont : références d'inflation
constante et saisonnière; ETS direct; Ridge IPCH; Ridge IPCH + PPI; ETS sur cinq
branches; saisonnier sur pays × 13 divisions; moyenne 50/50 ETS direct/branches.

La comparaison respecte la même grille de cibles et les horizons 1–3. Les
modèles directs utilisent le même début d'histoire que les cinq branches.
Le panel géographique utilise les historiques propres aux séries et ses replis
déclarés. Toutes les prévisions sont réalisées uniquement avec le passé.

Les rapports sont dans `reports/lab/runs/validation_.../` :

- `metrics.csv`, MAE/RMSE/biais et nombre de replis;
- `predictions.parquet`, toutes les prévisions et réalisations;
- `leaf_predictions.parquet`, prévisions détaillées pour l'incertitude;
- `metrics_by_period.csv`, `metrics_by_crosses_january.csv`;
- `rolling_rmse.csv`, `worst_errors_h3.csv`, `fit_failures.csv`;
- `experiment.json`, versions et empreintes;
- `origins/`, calculs intermédiaires permettant une reprise après interruption.

Une relance identique reprend les origines déjà calculées, ou réutilise le
résultat terminé. Une modification du code exige une nouvelle validation :

```bash
uv run python finish_project.py compare --restart
```

Les anciennes expériences sont conservées. Pour un choix explicite excluant
les PPI, `compare --without-ppi` retire cette variante de la comparaison; les
autres règles sont identiques.

**Passage :** grille complète; un échec reste visible et ne peut pas améliorer
le score par suppression de sa date.

## 5. Analyser erreurs, incertitude et sensibilité

```bash
uv run python finish_project.py analyze
uv run python finish_project.py compare --window 120
uv run python finish_project.py analyze --window 120
```

La première analyse porte sur la fenêtre croissante. La comparaison suivante
répète exactement le protocole avec dix années d'apprentissage. Elle a son
propre pointeur et conserve le résultat principal.

Les fichiers ajoutés aux rapports sont :

- `interval_predictions.csv`, intervalles calculés chronologiquement;
- `interval_metrics.csv`, couverture, largeur et score à 80 % et 95 %;
- `paired_loss_bootstrap.csv`, incertitude des différences de pertes à h=3;
- `residual_diagnostics.csv`, biais, autocorrélation et diagnostic descriptif;
- `analysis.json`, hypothèses de calibration.

La calibration exige au moins 36 origines complètes déjà arrivées à maturité.
Elle préserve les dépendances entre horizons et, pour le bottom-up, entre
cellules. Un élargissement causal est appliqué lorsque les douze dernières
erreurs sont plus volatiles. La couverture nominale reste à confronter à la
couverture observée, surtout pendant les crises.

Ouvrir ensuite `notebooks/02_model_comparison.ipynb` dans PyCharm, sélectionner
le kernel `.venv`, puis **Run All**. Le notebook lit les résultats : il ne
relance pas les modèles et ne consulte pas les cibles du test final.

**Passage :** choix d'une fenêtre et d'un modèle fondé sur la validation,
complété par la stabilité, les replis et les intervalles. Un petit écart de RMSE
dont l'intervalle de différence de pertes contient zéro ne démontre pas une
supériorité statistique.

## 6. Figer le choix avant le test final

Pour retenir le meilleur RMSE de validation avec la fenêtre croissante :

```bash
uv run python finish_project.py freeze
```

Si la fenêtre de dix ans est retenue après la comparaison précédente :

```bash
uv run python finish_project.py freeze --window 120
```

Ces deux commandes sont des alternatives. Sur l'extraction de vérification,
la moyenne ETS direct/branches avec dix ans obtient le meilleur RMSE parmi ces
deux comparaisons; c'est cette seconde commande qui a été utilisée pour le
test de la livraison. Consulter tes propres tableaux avant de choisir.

Pour choisir explicitement un candidat avant le test, par exemple ETS direct :

```bash
uv run python finish_project.py freeze --model direct__ets_additive
```

`reports/lab/selection.json` conserve le modèle, la fenêtre, les versions,
les empreintes et la règle de réestimation. Si une sélection a été enregistrée,
une modification explicite exige `freeze --replace` et reste possible seulement
avant la première ouverture du test.

**Passage :** sélection enregistrée et protocole figé. Cette commande n'ouvre
pas le test final.

## 7. Exécuter le test indépendant

```bash
uv run python finish_project.py test
```

La commande évalue la procédure figée et deux références sur octobre
2024–septembre 2026. Les modèles peuvent intégrer les nouvelles observations
au fur et à mesure, selon leur règle de réestimation; les familles, variables
et règles de sélection restent figées.

Les erreurs du test sont rapportées même si les résultats sont moins bons.
La calibration des intervalles peut incorporer les erreurs de test antérieures
déjà observées, conformément à la règle figée. Aucune erreur future n'entre
dans un intervalle passé.

L'ouverture est enregistrée dans `test_opened.json`. Une interruption se reprend
avec la même commande. Une nouvelle sélection après ouverture est bloquée.

**Passage :** résultats du test archivés, sans réajustement du choix à partir
de ses erreurs.

## 8. Produire les prévisions courantes

```bash
uv run python finish_project.py forecast
```

Les modèles sont réestimés jusqu'à septembre 2026. Le résultat porte sur
octobre, novembre et décembre 2026 et comprend les indices, les taux annuels,
les intervalles et la provenance.

Les fichiers sont :

- `reports/lab/current/forecast.csv`;
- `forecast_intervals.csv` et `forecast_metadata.json`;
- `reports/lab/current/dashboard.json`;
- `deploy/dashboard.json`, export autonome lu par le serveur.

La prévision détaillée pays × divisions est réconciliée vers la trajectoire
globale retenue pour servir de base cohérente aux chocs. Un scénario nul
retrouve exactement cette prévision centrale.

## 9. Exécuter et comprendre les scénarios

Exemple : hausse permanente de 10 % des prix de transport en France :

```bash
uv run python finish_project.py scenario --countries FR --components CP07 --magnitude 10 --kind permanent
```

Exemple : hausse de 5 % limitée à un mois, transport en France et Allemagne :

```bash
uv run python finish_project.py scenario --countries FR DE --components CP07 --magnitude 5 --kind temporary --duration 1
```

Exemple : supplément mensuel de 1 % pendant trois mois sur l'alimentation :

```bash
uv run python finish_project.py scenario --countries FR --components CP01 --magnitude 1 --kind monthly --duration 3
```

On peut préciser `--start 2026-11`; par défaut le choc commence au premier mois
prévu. La durée porte sur les hausses temporaires ou répétées. Les observations
antérieures restent inchangées. La sortie est limitée aux trois mois du cas.

Chaque exécution écrit `scenario.csv` et `decomposition.csv` dans
`reports/lab/scenarios/`. Les effets sont en **points de pourcentage**, distincts
du pourcentage de choc sur les prix. CP04 comprend tout le logement et les
combustibles : ce n'est pas un choc sur l'énergie seule. Les noms complets sont
affichés dans l'application.

Ouvrir `notebooks/03_scenarios.ipynb` pour comparer les profils et leurs effets.

## 10. Ouvrir le dashboard

```bash
uv run python finish_project.py serve
```

Ouvrir `http://127.0.0.1:8050`. Les quatre vues sont : vue d'ensemble,
modèles et validation, simulateur, méthodologie. La simulation a un bouton
de réinitialisation et un export CSV. Le serveur charge les résultats préparés;
il ne réentraîne pas les modèles à chaque clic.

Pour arrêter le serveur : **Ctrl + C** dans son terminal.

Le point `/healthz` renvoie 200 si le bundle de prévision est chargé, 503 si
la commande forecast n'a pas encore produit l'export. Une absence de données
ne déclenche pas de résultats fictifs.

Pour examiner uniquement l'export de vérification fourni, sans remplacer tes
résultats, depuis le projet :

```bash
CACIB_BUNDLE_PATH=example_results/dashboard.json uv run python finish_project.py serve
```

Cet export est identifié par sa propre collecte et sa coupure. Il ne remplace
pas l'exécution de ton backtest.

## 11. Vérifier et enregistrer le projet

Relancer les tests après une modification du code. Le dépôt contient un workflow
GitHub Actions pour les tests unitaires sans téléchargement de données.

```bash
uv run python -m unittest discover -p "test_*.py"
git add cacib_lab finish_project.py test_cacib_lab.py requirements-lab.txt GUIDE_COMPLETION.md METHODOLOGY.md notebooks/02_model_comparison.ipynb notebooks/03_scenarios.ipynb Dockerfile render.yaml pyproject.toml uv.lock
git add .github/workflows/cacib-lab-tests.yml .dockerignore
git add -f deploy/dashboard.json
git commit -m "Complete HICP evaluation, scenarios and interactive dashboard"
git push
```

Les données brutes et les caches volumineux restent hors du dépôt. La méthode de
téléchargement et les empreintes permettent de tracer les entrées. Refaire une
collecte plus tard ne garantit pas des observations identiques, à cause des
révisions d'Eurostat : conserver les instantanés pour reproduire exactement un
millésime.

## 12. Héberger sur Render ou lancer Docker

Le déploiement utilise le code Python et `deploy/dashboard.json`, sans lancer le
backtest sur le serveur. Dans Render, créer un **Blueprint**, relier le dépôt
GitHub et utiliser `render.yaml`. Le fichier configure Python 3.12, les
dépendances, Gunicorn, le port et `/healthz`. Choisir le nom de service désiré
avant de créer le Blueprint. Le plan est défini sur `free`; les déploiements
automatiques sont désactivés.

Pour une création manuelle de Web Service :

```text
Build command : pip install --no-cache-dir -r requirements-lab.txt
Start command : gunicorn cacib_lab.app:server --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 60
Health check  : /healthz
```

Variables : `PYTHON_VERSION=3.12.15`, `CACIB_BUNDLE_PATH=deploy/dashboard.json`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

Pour un lancement conteneurisé local, avec Docker installé :

```bash
docker build -t cacib-inflation-lab .
docker run --rm -p 8050:8050 cacib-inflation-lab
```

Le Dockerfile et la configuration Render sont fournis. Leur présence ne signifie
pas qu'un service a été publié : la création du service se fait dans ton compte
Render avec ton dépôt.

## Repérer les erreurs usuelles

| Message | Action |
|---|---|
| Module manquant | Relancer `uv add -r requirements-lab.txt` dans le projet |
| `inputs.json` absent | Exécuter `init` après préparation des données |
| PPI absents | Préparer les PPI avant init, ou utiliser explicitement `compare --without-ppi` |
| Extraction figée modifiée | Restaurer les fichiers de cette expérience; garder une nouvelle collecte dans un autre dossier |
| Code différent de la validation | Avant le test, relancer `compare --restart`, l'analyse et la sélection |
| Dashboard non prêt | Exécuter `forecast` après le test final |
| Test lié à une autre sélection | Conserver la sélection initiale; le test déjà consulté n'est plus indépendant pour un nouveau choix |

Le projet technique est terminé lorsque les tests passent, la comparaison et le
test indépendant sont archivés, le scénario nul retrouve la trajectoire centrale,
les hypothèses sont documentées et le dashboard fonctionne. La présentation est
volontairement exclue de cette livraison.
