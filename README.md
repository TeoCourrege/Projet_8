# Scoring API — Prêt à Dépenser

Mise en production du modèle de scoring crédit développé lors du projet
précédent ("Initiez-vous au MLOps") : une API FastAPI, sa conteneurisation
Docker, un pipeline CI/CD, et un monitoring de production avec détection de
dérive des données (data drift).

Le modèle servi est celui entraîné et suivi dans **MLflow** par le projet
précédent (`legacy_mlops_project/`, dataset Home Credit), exporté vers
`models/` par `scripts/export_mlflow_model.py`. Un modèle de démonstration
entraîné sur des **données synthétiques** reste disponible en secours (CI,
tests, conteneur sans modèle) — voir [Modèle de démonstration](#modèle-de-démonstration-synthétique).

## Sommaire

- [Architecture](#architecture)
- [Structure du projet](#structure-du-projet)
- [Démarrage rapide](#démarrage-rapide)
- [Utiliser l'API](#utiliser-lapi)
- [Tests](#tests)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Monitoring & data drift](#monitoring--data-drift)
- [Stockage des données de production](#stockage-des-données-de-production)
- [Historique des versions](#historique-des-versions)
- [Modèle de démonstration (synthétique)](#modèle-de-démonstration-synthétique)
- [Optimisation de l'inférence](#optimisation-de-linférence)
- [Limites](#limites)

## Architecture

```
legacy_mlops_project/run.py, run2.py ──▶ MLflow (mlflow.db : runs, métriques, modèles)
                                             │
                  scripts/export_mlflow_model.py (meilleur run → models/ + référence drift)
                                             ▼
Client HTTP ──▶ FastAPI (/predict) ──▶ pipeline sklearn (chargé une seule fois au démarrage)
                     │                        │
                     │                        └─▶ preprocessing + LightGBM calibré
                     ▼
        logs/predictions.jsonl (stockage local des inputs/outputs/latence)
                     │
                     ├─▶ scripts/run_drift_analysis.py  (PSI, KS-test vs. données de référence)
                     └─▶ dashboard/monitoring_app.py     (Streamlit : scores, latence, drift)
```

Le modèle est chargé **une seule fois**, au démarrage de l'API (`lifespan`
FastAPI), puis réutilisé pour toutes les requêtes — le recharger à chaque
appel serait beaucoup trop lent et gourmand en mémoire sous charge.

## Structure du projet

```
.
├── src/scoring_api/          # package de l'API
│   ├── main.py                # endpoints FastAPI
│   ├── schemas.py              # modèles Pydantic (validation d'entrée)
│   ├── features.py             # mapping API ↔ colonnes Home Credit + feature engineering
│   │                           # (partagé par l'API et legacy_mlops_project/utils.py)
│   ├── model.py                 # chargement du modèle / inférence
│   ├── fast_inference.py        # inférence optimisée (pipeline compilé au démarrage)
│   ├── config.py                 # configuration (variables d'environnement)
│   ├── logging_utils.py           # logging structuré des prédictions
│   └── monitoring/drift.py         # PSI / KS-test pour la détection de drift
├── scripts/
│   ├── export_mlflow_model.py      # meilleur run MLflow -> models/ (+ Model Registry)
│   ├── generate_synthetic_data.py  # données synthétiques (modèle de démo)
│   ├── train_model.py               # modèle de démo -> models/
│   ├── simulate_traffic.py           # rejoue des demandes réelles (option --drift)
│   ├── benchmark_inference.py        # profiling + benchmark avant/après optimisation
│   └── run_drift_analysis.py          # rapport de drift (CSV + JSON)
├── dashboard/monitoring_app.py         # dashboard Streamlit
├── tests/                                # pytest (API + feature engineering)
├── legacy_mlops_project/                  # entraînement MLflow (projet précédent)
├── Dockerfile, docker-compose.yml, docker/entrypoint.sh
├── .github/workflows/ci-cd.yml             # pipeline CI/CD (tests, build, déploiement simulé)
└── docs/screenshots/                        # captures d'écran du stockage de prod (à ajouter)
```

## Démarrage rapide

### Prérequis

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) (recommandé) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker (optionnel, pour la conteneurisation)

### Installation

Avec **uv** (recommandé) :

```bash
uv sync --extra dev --extra monitoring --extra training
```

(`training` = MLflow + matplotlib, uniquement pour l'entraînement et l'export ;
l'image Docker de l'API ne l'installe pas.)

Ou avec **pip** :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Entraîner (MLflow) et exporter le modèle

Prérequis : les CSV Home Credit dans `data/raw/` (`application_train.csv`,
`application_test.csv`).

```bash
cd legacy_mlops_project
uv run python run.py      # sélection de modèles (LightGBM, RandomForest, MLP...)
uv run python run2.py     # (optionnel) grid search LightGBM
cd ..
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # comparer les runs

uv run python scripts/export_mlflow_model.py --register
```

Les runs n'utilisent que les colonnes que l'API collecte
(`scoring_api.features.HC_RAW_COLUMNS`) : le modèle loggé est donc servable
tel quel. L'export choisit le meilleur run terminé (`oof_roc_auc` max, ou
`--run-id <id>` / `--metric <nom>`), vérifie que ses colonnes d'entrée
correspondent à celles de l'API, puis écrit :

- `models/pipeline.joblib` — le pipeline sklearn (preprocessing + modèle),
- `models/metadata.json` — version, run MLflow, **seuil de décision**
  (`best_threshold` du run, issu de la fonction de coût 10×FN + FP), métriques,
- `data/processed/reference_sample.csv` — 20 000 lignes d'entraînement au
  format de l'API, référence pour la détection de drift.

`--register` enregistre aussi le modèle dans le Model Registry MLflow
(`credit-scoring`, alias `champion`).

### Configuration (optionnel)

Toutes les options ont des valeurs par défaut sensées. Pour les modifier,
copiez `example.env` vers `.env` (ignoré par git) et ajustez :

```bash
cp example.env .env
```

### Lancer l'API

```bash
uv run uvicorn scoring_api.main:app --reload
```

- Documentation interactive (Swagger) : http://127.0.0.1:8000/docs
- Health check : http://127.0.0.1:8000/health

### Générer du trafic de démonstration (pour le monitoring)

Le simulateur rejoue des demandes réelles tirées de
`data/raw/application_test.csv` (clients jamais vus à l'entraînement) :

```bash
uv run python scripts/simulate_traffic.py --n-requests 300
# puis, pour simuler une dérive de population :
uv run python scripts/simulate_traffic.py --n-requests 150 --drift
```

### Lancer le dashboard de monitoring

```bash
uv run streamlit run dashboard/monitoring_app.py
```

### Lancer l'analyse de drift (script/rapport)

```bash
uv run python scripts/run_drift_analysis.py
```

Génère `monitoring_reports/feature_drift.csv` et `monitoring_reports/monitoring_summary.json`.

## Utiliser l'API

`POST /predict` — reçoit les données d'un client, retourne un score et une décision :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "F", "own_car": true, "own_realty": true,
    "contract_type": "Cash loans", "children": 1, "family_members": 3,
    "income_total": 65000, "credit_amount": 180000, "annuity_amount": 12000,
    "goods_price": 175000, "age_years": 35, "years_employed": 6,
    "education_type": "Higher education", "family_status": "Married",
    "housing_type": "House / apartment", "occupation_type": "Core staff",
    "organization_type": "Business Entity Type 3",
    "ext_source_1": 0.6, "ext_source_2": 0.55, "ext_source_3": 0.5
  }'
```

Réponse :

```json
{
  "request_id": "…",
  "score": 0.1834,
  "decision": "ACCEPTE",
  "threshold": 0.42,
  "model_version": "LightGBM-48c8e22e",
  "latency_ms": 4.2,
  "timestamp": "2026-01-01T12:00:00+00:00"
}
```

`score` est la probabilité de défaut estimée ; `REFUSE` si `score >= threshold`.

Autres endpoints : `GET /health`, `GET /model/info`, `GET /docs`.

### Validation des entrées

Toute entrée invalide renvoie un `422` avec le détail de l'erreur : champ
obligatoire manquant, valeur hors plage (âge négatif, revenu ≤ 0, annuité
supérieure au crédit...), type incorrect (texte au lieu d'un nombre),
catégorie inconnue. Voir `src/scoring_api/schemas.py` et `tests/test_api.py`.

Les catégories acceptées sont exactement celles du dataset Home Credit.
`occupation_type`, `organization_type`, `years_employed` et `ext_source_*`
sont optionnels. `years_employed: null` (retraité / sans emploi) correspond au
`DAYS_EMPLOYED = 365243` du dataset, et `organization_type: null` à `XNA`.
L'API convertit ces champs en colonnes Home Credit (`age_years` →
`DAYS_BIRTH`...) via `scoring_api.features.to_home_credit_frame`.

## Tests

```bash
uv run pytest -v
```

Les tests entraînent un petit modèle de démo à la volée (fixture de session
dans `tests/conftest.py`) : ils sont donc reproductibles sans dépendre des
scripts CLI ni du dataset complet.

## Docker

```bash
docker build -t scoring-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" -v "$(pwd)/logs:/app/logs" scoring-api
```

Si aucun modèle n'est présent dans `models/` au démarrage du conteneur,
`docker/entrypoint.sh` génère automatiquement des données synthétiques et
entraîne un modèle de démonstration avant de lancer l'API — l'image est donc
utilisable telle quelle, sans préparation préalable.

Avec `docker-compose` (API + dashboard de monitoring) :

```bash
docker compose up --build
```

- API : http://localhost:8000
- Dashboard : http://localhost:8501

> Ces commandes lancent uniquement des conteneurs **locaux** — rien n'est
> déployé ni exposé publiquement.

## CI/CD

`.github/workflows/ci-cd.yml` s'exécute sur chaque push/PR vers `main` et
enchaîne trois jobs :

1. **test** — lint (`ruff`) + tests (`pytest --cov`).
2. **build** — construit l'image Docker avec le modèle exporté de MLflow
   s'il est versionné dans `models/`, sinon avec un modèle de démo
   entraîné à la volée.
3. **deploy** — déploiement simulé localement dans le runner : lance le
   conteneur, attend `/health`, exécute un smoke test sur `/predict`, puis
   l'arrête.

## Monitoring & data drift

Le dashboard Streamlit (`dashboard/monitoring_app.py`) affiche :

- nombre de requêtes traitées, taux de refus,
- latence moyenne / p95 de l'API,
- distribution des scores prédits (référence vs. production),
- répartition des décisions (ACCEPTE / REFUSE),
- latence par requête dans le temps,
- **dérive des données** par variable, mesurée par le **PSI** (Population
  Stability Index — standard en scoring crédit) et un test de
  Kolmogorov-Smirnov pour les variables numériques.

Grille de lecture du PSI (`src/scoring_api/monitoring/drift.py`) :

| PSI | Interprétation |
|---|---|
| < 0.10 | stable |
| 0.10 – 0.25 | drift modéré — à surveiller |
| > 0.25 | drift significatif — ré-entraînement à envisager |

Ce projet implémente PSI/KS "à la main" (pas de dépendance à Evidently ou
NannyML) pour rester léger et stable dans le temps ; ces bibliothèques
restent des alternatives valables si vous préférez un rapport HTML tout fait.

## Stockage des données de production

En local, chaque appel à `/predict` est journalisé en JSON Lines dans
`logs/predictions.jsonl` (`src/scoring_api/logging_utils.py`) : timestamp,
input complet, score, décision, latence. C'est le PoC "stockage local"
explicitement autorisé par la mission.

En production réelle, on remplacerait ce fichier par une solution managée
(Elasticsearch/OpenSearch pour la recherche de logs, PostgreSQL pour des
requêtes analytiques, ou un object store type S3 pour l'archivage brut).

**Captures d'écran** : ajoutez vos captures de la solution de stockage
(contenu de `logs/predictions.jsonl`, dashboard, ou service cloud choisi)
dans `docs/screenshots/`.

## Historique des versions

L'historique de commits (init Git, ajout de l'API, du Dockerfile, du
pipeline CI/CD, du monitoring...) est à consulter directement sur le dépôt
GitHub une fois poussé — voir `git log` en local.

## Modèle de démonstration (synthétique)

Sans le dataset Home Credit (CI, tests, conteneur vierge), un modèle de démo
au même contrat d'entrée peut être entraîné sur des données synthétiques :

```bash
uv run python scripts/generate_synthetic_data.py   # data/raw/credit_applications.csv
uv run python scripts/train_model.py               # models/ + référence drift
```

Il ne sert qu'à faire tourner la chaîne de bout en bout : ses scores n'ont
pas de valeur métier.

## Optimisation de l'inférence

Rapport complet : [`docs/optimisation_report.md`](docs/optimisation_report.md).

Le profiling (`cProfile`) a montré que ~80 % du temps d'inférence venait de
pandas et de sklearn appliqués à une seule ligne, pas du modèle. L'API
« compile » donc le pipeline au démarrage (`src/scoring_api/fast_inference.py`) :
features en Python pur, prétraitement en NumPy, appel direct des arbres
LightGBM sur un thread.

| | Avant | Après |
|---|---|---|
| Inférence p50 (conteneur Docker) | 7,3 ms | 0,08 ms (×90) |
| Temps de réponse API p50 | 14,2 ms | 6,7 ms |
| Scores / décisions (19 979 clients) | — | identiques (écart 0,0) |

ONNX Runtime a été testé et écarté : il change les scores (calibration
différente). Au démarrage, l'API vérifie que le chemin optimisé donne
exactement les scores du pipeline sklearn et revient à sklearn sinon ; le
moteur utilisé est visible dans `GET /model/info` (`inference_engine`).
Désactivation : `SCORING_API_FAST_INFERENCE=false`.

Reproduire les mesures :

```bash
uv run python scripts/benchmark_inference.py
```

## Limites

- Le logging fichier (`logs/predictions.jsonl`) n'est pas process-safe pour
  plusieurs workers Uvicorn en parallèle ; en prod, router les logs vers un
  système centralisé (cf. section stockage) lève cette limite.
- Après optimisation, le temps de réponse est dominé par la couche HTTP
  (~6 ms) : pistes dans le rapport d'optimisation.
