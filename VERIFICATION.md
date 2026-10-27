# Vérification de la livraison

Les résultats ci-dessous proviennent de l'extraction de vérification du projet,
avec coupure septembre 2026. Les valeurs de ton exécution dépendent de ton
instantané Eurostat et de l'environnement verrouillé dans uv.lock.

## Exécutions terminées

- 40 tests des modules précédents : réussis.
- 42 nouveaux tests unitaires, temporels, mathématiques et HTTP : réussis.
- 3096 prévisions de validation avec fenêtre croissante : aucun échec.
- 3096 prévisions de validation avec fenêtre de 120 mois : aucun échec.
- 216 prévisions sur le test final avec les trois procédures figées : aucun échec.
- 8 cellules de code du notebook 02 et 7 du notebook 03 exécutées sans erreur.
- Routes du dashboard `/`, `/healthz`, `/_dash-layout`, `/_dash-dependencies`
  et feuille de style : réponses 200 avec l'export réel.
- Callback de scénario et export CSV : vérifiés par requêtes HTTP Dash.
- Scénarios réels multi-pays et multi-catégories : impacts et attribution cohérents.

## Résultats à trois mois

| Procédure | RMSE validation croissante | RMSE validation 120 mois |
|---|---:|---:|
| ETS direct | 0.740 | 0.711 |
| ETS cinq branches | 0.745 | 0.734 |
| Moyenne 50/50 des deux ETS | 0.721 | 0.708 |
| Ridge IPCH | 0.790 | 0.866 |
| Ridge IPCH + PPI | 0.764 | 0.780 |
| Saisonnier pays × 13 divisions | 0.844 | 0.844 |

La procédure retenue avant ouverture du test est la moyenne 50/50, fenêtre
120 mois. Sur octobre 2024–septembre 2026 :

| Procédure figée | MAE | RMSE |
|---|---:|---:|
| Inflation annuelle constante | 0.390 | 0.532 |
| Moyenne ETS retenue | 0.397 | 0.539 |
| ETS direct | 0.447 | 0.580 |

Erreurs en points de pourcentage. La référence d'inflation constante est donc
légèrement meilleure sur cette période finale. Ce résultat est conservé;
il n'a pas servi à modifier le choix figé. Les intervalles de différences de
pertes restent à interpréter avec le nombre limité de cibles et la sélection
de plusieurs candidats.

## Limites de la vérification technique

Les cellules des notebooks ont été exécutées avec IPython dans le processus de
vérification. Le lancement d'un kernel Jupyter séparé n'était pas possible dans
cet environnement, car les sockets TCP et IPC étaient indisponibles. Les
notebooks restent au format Jupyter standard, avec leurs sorties enregistrées
et cette méthode de vérification inscrite dans leurs métadonnées.

Le rendu visuel dans un navigateur n'a pas été vérifié : l'environnement
d'aperçu ne disposait pas de l'application Python et de ses dépendances. Les
routes et callbacks ont été testés avec le client HTTP du serveur Flask réel.

Docker n'était pas disponible pour construire l'image. Le Dockerfile et la
configuration Render sont fournis et documentés, sans déploiement effectué
dans un compte externe.

Les résultats de référence sont dans `example_results/`; les données brutes
et les caches d'apprentissage volumineux ne sont pas inclus dans l'archive.
