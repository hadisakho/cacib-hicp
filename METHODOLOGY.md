# Méthodologie de la suite CA-CIB

## Cible et protocole

L'objectif principal est l'inflation IPCH annuelle de la zone euro à trois mois.
Les horizons 1 et 2 sont également évalués. Pour h <= 3 :

\[
\widehat\pi_{t+h|t}=100\left(\frac{\widehat I_{t+h|t}}{I_{t+h-12}}-1\right).
\]

Le dénominateur annuel est déjà observé à l'origine. Les modèles Ridge estiment
directement la cible annuelle pour chaque horizon, puis sont reconvertis en
indices positifs; ETS et les références prévoient les niveaux.

Le protocole figé est : validation des cibles janvier 2014–septembre 2024;
test octobre 2024–septembre 2026; coupure septembre 2026. Une actualisation des
données ne décale pas automatiquement ces frontières. L'histoire commune de
TOTAL et des cinq branches commence au premier mois disponible pour toutes.
Les approches pays utilisent les historiques propres aux séries, avec une règle
de repli identique à toutes les origines. La fenêtre principale est croissante;
la sensibilité utilise 120 mois, avec le même minimum requis de 120 mois.

Chaque prévision reçoit uniquement les observations jusqu'au mois d'origine.
La cible est l'EA à composition évolutive publiée par Eurostat. Il s'agit d'un
backtest rétrospectif sur un millésime téléchargé, pas d'une reconstitution des
publications historiques. Les anciennes valeurs peuvent avoir été révisées ou
reclassées rétrospectivement sous ECOICOP v2.

## Candidats bornés

1. Inflation annuelle constante.
2. Profil saisonnier moyen de cinq années passées.
3. ETS additif avec tendance amortie et saisonnalité mensuelle, module existant.
4. Ridge avec retards IPCH, effet de base connu et mois cible.
5. Même Ridge avec six variations PPI : énergie, biens intermédiaires,
   alimentation/boissons/tabac, chaque fois variation mensuelle et annuelle.
6. ETS sur cinq branches puis agrégation.
7. Profil saisonnier sur pays × 13 divisions puis agrégation.
8. Moyenne arithmétique 50/50 des indices prévus par ETS direct et ETS branches.

SARIMA et le boosting ont été étudiés dans les étapes précédentes. Ils ne sont
pas relancés par défaut dans ce comparatif de finalisation. Aucune PCA n'est
intégrée. Le modèle saisonnier détaillé limite le coût du panel : il constitue
un comparateur et une base transparente pour les scénarios, pas une hypothèse
de supériorité du bottom-up. Les huit candidats sont déclarés avant la lecture
du test final.

Ridge choisit alpha dans la grille existante [0.1, 1, 10, 100, 1000] avec trois
plis chronologiques de douze mois. Le gap h-1 empêche un label d'apprentissage
de dépasser la première origine de validation interne. StandardScaler est
ajusté dans chaque pli d'apprentissage. Pour chaque origine historique s, les
variables externes sont déjà construites à partir de PPI(s-2) et du passé.
Ce décalage prudent ne remplace pas les véritables dates de publication.
Les PPI EA20 sont des indicateurs explicatifs d'un périmètre fixe, distincts
de la cible EA évolutive.

## Qualité et replis

Les tables sources et les statuts Eurostat sont conservés. Un indice nul ou
non positif est exclu de la vue d'apprentissage; il n'est pas interprété comme
un prix gratuit et n'est pas modifié dans la source. Le modèle utilise le
dernier segment continu de valeurs positives jusqu'à l'origine.

Dans le panel désagrégé seulement, si le modèle ne peut pas être ajusté, le repli
est : inflation annuelle constante si les treize derniers mois sont complets;
sinon dernier niveau positif observé avant l'origine. Une série jamais observée
à poids positif bloque l'agrégation. Une série inactive sans historique reste
absente. Les pays qui ne sont pas encore membres peuvent avoir des prévisions
de calibration à partir de leur passé; leur contribution EA est nulle.

Une base de décembre manquante peut être remplacée, pour le chaînage seulement,
par sa dernière valeur positive antérieure. Ce report causal est journalisé;
l'histoire utilisée par le modèle n'est pas interpolée. Les audits descriptifs
de reconstruction utilisent les observations brutes et signalent toute
reconstruction impossible, sans remplir les lacunes.

## Agrégation et géographie

Dans une année y, chaque indice est rapporté au décembre précédent. Ces rapports
sont agrégés avec les poids annuels puis multipliés par le niveau agrégé de
décembre précédent. Pour les pays, le poids d'une cellule est le produit du
poids pays et du poids de la division dans ce pays, chacun normalisé par sa
somme publiée. Les sommes de poids initiales doivent être proches de 1000
(tolérance 0.2 liée aux arrondis). Aucun panier incomplet n'est renormalisé
après suppression de catégories à poids positif.

Le diagnostic historique applique la composition et les poids de chaque année.
Une prévision utilise les poids et les membres de l'année de son origine,
reconduits si l'horizon traverse janvier. Ainsi, une origine de décembre
précédant une adhésion ne prétend pas connaître les futurs poids du nouvel
ensemble. Cet écart avec une information historique parfaitement reconstituée
est explicite. La prévision courante de septembre 2026 utilise la composition
et les poids observés en 2026.

La reconstruction descriptive utilise une ancre officielle à chaque décembre.
Elle ne doit pas être confondue avec un indice entièrement reconstruit depuis
une unique ancre initiale. Les erreurs d'arrondi sont mesurées séparément des
erreurs de prévision.

## Évaluation, sélection et test

La grille contient exactement chaque modèle × horizon × mois cible prévu.
Un échec reste une ligne de la grille et rend son score global inéligible;
il n'est pas supprimé pour améliorer le classement. Les replis désagrégés font
partie de la procédure évaluée et sont comptabilisés dans les rapports.

RMSE à h=3 est le critère principal, avec MAE, biais, résultats par période,
franchissement de janvier et RMSE glissant de douze mois. Le bootstrap apparié
des différences de pertes quadratiques utilise des blocs de six mois et
2000 tirages. Il conserve une partie de la dépendance temporelle, notamment
pour les horizons qui se chevauchent; les intervalles restent descriptifs et
ne corrigent pas automatiquement la multiplicité de tous les modèles testés.

`freeze` choisit le minimum de RMSE de validation, ou un candidat explicitement
nommé avant le test, puis enregistre code, versions, données et procédure.
`test` verrouille cette sélection dès sa première ouverture, même si une
exécution est interrompue. Les paramètres peuvent être réestimés sur le passé
de chaque origine de test; les résultats ne servent pas à sélectionner une
autre famille ou de nouvelles variables. Le test contient 24 cibles par
horizon : son incertitude d'évaluation doit être prise en compte.

## Incertitude

Les intervalles sont produits à partir d'erreurs hors échantillon. À l'origine
t, une erreur n'entre dans la calibration que si sa cible est <= t. Une ligne
de calibration conjointe exige les trois horizons observés. Au moins 36
origines complètes sont nécessaires; les premières dates ne reçoivent donc
pas d'intervalles évaluables. Leur absence est explicite.

Pour le direct et la combinaison, un tirage sélectionne tout le vecteur des
trois erreurs logarithmiques de niveau d'une même origine. Pour le bottom-up,
le tirage sélectionne conjointement les erreurs des cellules positives et des
trois horizons, construit les trajectoires d'indices puis les agrège. Cela
préserve les dépendances empiriques entre cellules et horizons, sans additionner
des bornes d'intervalles. Si la calibration détaillée complète est insuffisante,
le moteur utilise les vecteurs d'erreurs agrégées et enregistre ce repli.

Un facteur causal par horizon élargit les erreurs simulées lorsque leur RMS sur
les douze dernières origines matures dépasse leur RMS historique. Le facteur
vaut max(1, RMS récent / RMS historique), sans rétrécissement automatique après
une accalmie. Il s'applique conjointement aux cellules d'un même horizon pour
préserver leurs dépendances. Ce choix est fixé sur la validation avant le test;
la couverture nominale reste un objectif à évaluer, pas une garantie.

Aux extrémités où seule une partie de la trajectoire est dans la phase évaluée,
la calibration agrégée utilise les bonnes colonnes d'horizon. La couverture,
la largeur et le score d'intervalle sont évalués chronologiquement, séparément
à 80 % et 95 %. Ce sont des intervalles empiriques sous une hypothèse de
stabilité locale des erreurs, sans garantie universelle, notamment en crise.

## Scénarios

Le moteur applique le choc aux indices des catégories choisies, jamais
directement au taux annuel global. Une hausse permanente multiplie le niveau
par 1+s à partir du début. Une hausse temporaire applique ce facteur pendant
la durée, puis revient au niveau central de la catégorie. Une hausse mensuelle
répétée multiplie par (1+s)^k, jusqu'à la durée choisie, puis conserve le niveau
atteint. Un taux s <= -100 % est interdit. Une catégorie ne reçoit pas deux
chocs superposés sans combinaison explicite.

Au changement d'année, le chaînage et les poids maintenus fixes peuvent laisser
un résidu agrégé après la fin d'un choc temporaire, même lorsque les catégories
retrouvent leur niveau central. Le moteur recalcule cette conséquence au lieu
de forcer artificiellement un impact nul.

Si le meilleur modèle est direct, les trajectoires pays × divisions sont
réconciliées par un facteur proportionnel positif, mois par mois, vers sa
prévision globale. Cette allocation conditionnelle préserve leurs prix relatifs
à chaque horizon; elle n'est pas une estimation causale de la transmission.
L'attribution séquentielle des effets somme exactement à l'impact total; en
présence d'interactions son ordre est explicite et influence la répartition.
Les intervalles centraux ne sont pas présentés comme des intervalles de
confiance causaux du scénario.

## Références primaires

- Eurostat, procédure de chaînage et agrégation géographique :
  https://ec.europa.eu/eurostat/en/web/hicp/methodology/
- Eurostat, classifications, arrondis, poids et calendrier :
  https://ec.europa.eu/eurostat/web/hicp/information-data
- scikit-learn, prévention des fuites de données et pipelines :
  https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn, TimeSeriesSplit :
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Dash et Dash Mantine Components :
  https://dash.plotly.com/ et https://www.dash-mantine-components.com/
- Render, services Python et configuration :
  https://render.com/docs/web-services et https://render.com/docs/blueprint-spec
