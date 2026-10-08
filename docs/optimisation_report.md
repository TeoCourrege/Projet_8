# Rapport d'optimisation post-déploiement

Modèle : `LightGBM-b8464ecf` (run MLflow exporté, LightGBM calibré par isotonie,
3 sous-modèles). Mesures brutes : `docs/optimisation/benchmark_results.json`
(Mac, arm64, 8 CPU) et `docs/optimisation/docker/benchmark_results.json`
(conteneur Docker de production, Linux aarch64, 2 CPU). Profils `cProfile` :
`docs/optimisation/**/profile_*.txt`.

Reproduire :

```bash
uv run python scripts/benchmark_inference.py                     # local, + API de bout en bout
uv run --with skl2onnx --with onnxmltools --with onnxruntime \
    python scripts/benchmark_inference.py --onnx                 # + essai ONNX Runtime
```

## 1. Point de départ : ce que montre le monitoring

Les logs de production (`logs/predictions.jsonl`, champ `latency_ms`) donnaient
un temps d'inférence de **~9 ms par requête** (p50), pour un temps de réponse
de l'API de **~14 ms**. Le modèle (des arbres de décision sur une seule ligne)
devrait pourtant se compter en dixièmes de milliseconde : l'hypothèse était un
surcoût hors modèle.

## 2. Profiling : où part le temps (`cProfile` + chronométrage par étape)

| Étape (p50, chemin d'origine) | Mac | Docker | Part |
|---|---|---|---|
| Features avec pandas (`to_home_credit_frame` + `engineer_features`) | 5,21 ms | 3,82 ms | ~55-60 % |
| Prétraitement sklearn (`ColumnTransformer` : imputation, normalisation, one-hot) | 1,97 ms | 1,82 ms | ~25 % |
| Modèle (`CalibratedClassifierCV.predict_proba`) | 1,33 ms | 1,29 ms | ~15 % |
| **Total inférence** | **8,83 ms** | **7,29 ms** | |

**Goulots d'étranglement identifiés** (cf. `profile_baseline.txt`) :

1. **pandas sur une seule ligne** : 26 insertions de colonnes (`DataFrame.__setitem__`,
   ~7 800 appels pour 300 requêtes) et des `where` ; pandas est fait pour des
   colonnes de millions de lignes, pas pour une ligne.
2. **Validation sklearn répétée** : `check_array` est appelé ~31 fois par requête
   (`ColumnTransformer`, chaque sous-pipeline, chaque sous-modèle calibré).
3. **Le modèle lui-même** ne représente que ~15 % du temps.

## 3. Stratégies testées

| Stratégie | Résultat | Retenue |
|---|---|---|
| LightGBM sur 1 thread au lieu de tous les CPU (`num_threads=1`) | appel des arbres 0,087 → 0,070 ms (−20 %) : lancer des threads coûte plus cher que prédire une ligne | ✅ |
| Features calculées en Python pur à partir de la requête validée | 5,2 ms → quelques µs | ✅ |
| Prétraitement « compilé » en NumPy (médianes, moyennes/écarts-types et positions one-hot extraites une fois au démarrage) | 1,97 ms → 0,005 ms (avec l'étape précédente) | ✅ |
| Appel direct des 3 boosters LightGBM + interpolation isotonique (sans la couche sklearn) | 1,33 ms → 0,075 ms | ✅ |
| **ONNX Runtime** (conversion `skl2onnx` + `onnxmltools` du modèle calibré) | étape modèle 0,020 ms, **mais scores faux** : écart max 0,39, **621 décisions différentes** sur 1 000 | ❌ |

**Pourquoi ONNX est écarté.** Avec LightGBM 4.x, sklearn calibre le **score brut**
(`decision_function`) du modèle, alors que le convertisseur ONNX calibre la
**probabilité** : la conversion produit donc un autre modèle. C'est exactement la
régression de précision à éviter. Le gain restant (~0,05 ms) serait de toute façon
négligeable devant le temps de réponse HTTP, pour une dépendance de 20 Mo en plus.
La même erreur a d'ailleurs été détectée et corrigée dans le chemin optimisé (appel
des boosters avec `raw_score=True`) grâce au contrôle d'équivalence.

## 4. Résultats

### Temps d'inférence (1 000 requêtes réelles de l'échantillon de référence)

| | Avant (sklearn) | Après (optimisé) | Gain |
|---|---|---|---|
| Mac — p50 / p95 | 8,83 / 9,55 ms | 0,083 / 0,092 ms | **×107** |
| Docker (prod) — p50 / p95 | 7,29 / 8,37 ms | 0,081 / 0,091 ms | **×90** |

### Temps de réponse de l'API (serveur uvicorn réel, 650 requêtes)

| | Avant | Après |
|---|---|---|
| p50 | 14,2 ms | **6,7 ms** (−53 %) |
| p95 | 15,9 ms | **8,9 ms** (−44 %) |

### Aucune régression de précision (19 979 clients de référence)

| | Avant | Après |
|---|---|---|
| Écart maximal de score | — | **0,0** (identique au bit près) |
| Décisions différentes | — | **0** |
| Taux de refus | 28,014 % | 28,014 % |
| ROC AUC | 0,784094 | 0,784094 |

Aucun biais introduit : les scores étant identiques pour chaque client, aucune
population n'est traitée différemment.

## 5. Mise en œuvre et sécurité du déploiement

- Code : `src/scoring_api/fast_inference.py`. Le pipeline est « compilé » **une
  seule fois au démarrage** de l'API, comme le chargement du modèle.
- **Contrôle d'équivalence au démarrage** : l'API compare le chemin optimisé au
  pipeline sklearn sur des clients types (champs optionnels absents, client sans
  emploi…). Au moindre écart (> 1e-9), ou si un futur modèle a une autre
  structure (MLP, RandomForest…), elle revient automatiquement au chemin sklearn :
  un nouvel export MLflow ne peut pas casser l'API.
- Pour un modèle non-LightGBM, seul le prétraitement est optimisé
  (`optimized-preprocessing`), ce qui supprime déjà ~80 % du temps.
- Interrupteur de retour arrière : `SCORING_API_FAST_INFERENCE=false`.
- Visibilité : `GET /model/info` et chaque ligne de log indiquent le moteur
  utilisé (`inference_engine`), ce qui permet de comparer avant/après dans le
  monitoring.
- Tests : `tests/test_fast_inference.py` (scores identiques sur ~400 clients,
  repli automatique, interrupteur). La CI vérifie après déploiement que le
  conteneur tourne bien en mode optimisé (`/model/info`).

## 6. Configuration finale retenue

- **Bibliothèques** : aucune nouvelle dépendance. NumPy et LightGBM (déjà
  présents) suffisent ; ONNX Runtime écarté (§3). L'image Docker ne change pas
  de taille.
- **Logiciel** : 1 worker uvicorn, LightGBM sur 1 thread par prédiction. Pour
  monter en charge, il vaut mieux ajouter des workers/conteneurs (un par CPU)
  que des threads par prédiction.
- **Matériel** : CPU uniquement. Un GPU serait inutile : l'inférence ne prend
  plus que ~0,08 ms. Les mesures Docker (2 CPU) montrent que le gain tient sur
  une petite machine.

## 7. Nouveau goulot et pistes suivantes

L'inférence ne représente plus que ~0,1-0,3 ms sur les ~6,7 ms du temps de
réponse : le goulot est désormais la couche HTTP (sérialisation JSON,
validation Pydantic, exécution de l'endpoint synchrone dans un thread,
écriture du log sur disque à chaque requête). Pistes, si le besoin de latence
le justifie : endpoint `async`, écriture des logs en tâche de fond ou par lots,
plusieurs workers uvicorn.
