# Projet Data Science — Tarification Assurance Auto

Ce projet construit un moteur de tarification en assurance automobile à partir de données historiques de contrats et de sinistres.

L'objectif est d'estimer le coût attendu des sinistres afin de calculer une prime cohérente et techniquement justifiée, selon la décomposition actuarielle classique :

```
Prime = Probabilité de sinistre × Coût moyen conditionnel au sinistre
```

---

## Structure du projet

```
Projet_data_science_assurance/
│
├── Frequence/                        # Modèle de fréquence (probabilité de sinistre)
│   ├── models/                       # Pickle généré par main.py --mode train
│   │   └── catboost_calibrated.pkl
│   ├── outputs/                      # Résultats de prédiction (non versionnés)
│   │   └── proba_sinistre.csv
│   └── src/
│       ├── main.py                   # Point d'entrée (entraînement + prédiction)
│       ├── preprocessing.py          # Nettoyage et feature engineering
│       ├── train.py                  # Entraînement, calibration, sauvegarde pickle
│       └── predict.py                # Chargement pickle et génération des probabilités
│
├── Severite/                         # Modèle de sévérité (coût moyen du sinistre)
│   ├── models/                       # Pickles générés par les scripts d'entraînement
│   │   ├── catboost_severite.pkl
│   │   └── model_rf_poids.pkl
│   ├── outputs/                      # Résultats de prédiction (non versionnés)
│   │   └── pred_severite.csv
│   └── src/
│       ├── main.py                   # Point d'entrée (entraînement + prédiction)
│       ├── preprocessing.py          # Nettoyage, imputation poids, feature engineering
│       ├── train.py                  # Entraînement CatBoost et sauvegarde pickle
│       ├── predict.py                # Chargement pickle et génération des prédictions
│       └── train_rf_poids.py         # ⚠️ À lancer une seule fois avant train.py
│
├── Prime/                            # Calcul de la prime finale
│   ├── outputs/                      # Résultats (non versionnés)
│   │   └── primes.csv
│   └── src/
│       └── main.py                   # Combine fréquence × sévérité → prime
│
├── data/                             # Données brutes (non versionnées)
│   ├── train.csv
│   └── test.csv
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/RaphaelCalhegas/Projet_data_science_assurance.git
cd Projet_data_science_assurance
pip install -r requirements.txt
```

Vérifier que les données sont présentes :
```
data/train.csv
data/test.csv
```

---

## Utilisation

### Étape 1 — Entraîner le modèle de poids (une seule fois)

Le modèle de sévérité utilise un Random Forest pour imputer les poids manquants des véhicules. Ce pickle doit être généré avant tout entraînement.

```bash
cd Severite/src
python train_rf_poids.py
```

→ Sauvegarde `Severite/models/model_rf_poids.pkl`

---

### Étape 2 — Entraîner le modèle de fréquence (une seule fois)

```bash
cd Frequence/src
python main.py --mode train
```

→ Sauvegarde `Frequence/models/catboost_calibrated.pkl`

---

### Étape 3 — Entraîner le modèle de sévérité (une seule fois)

```bash
cd Severite/src
python main.py --mode train
```

→ Sauvegarde `Severite/models/catboost_severite.pkl`

---

### Étape 4 — Calculer les primes

```bash
cd Prime/src
python main.py

# Sur un fichier spécifique
python main.py --data ../../data/test.csv
```

→ Exporte `Prime/outputs/primes.csv` avec `index` et `prime` par contrat

---

### Générer les prédictions individuelles (optionnel)

```bash
# Probabilités de sinistre
cd Frequence/src
python main.py --mode predict

# Montants de sinistre
cd Severite/src
python main.py --mode predict
```

---

## Description des fichiers

| Fichier | Rôle |
|---|---|
| `Frequence/src/main.py` | Orchestre entraînement et prédiction fréquence via `--mode` |
| `Frequence/src/preprocessing.py` | Valeurs manquantes, tranches d'âge, extraction département |
| `Frequence/src/train.py` | Entraîne CatBoostClassifier, calibre (isotonic), sauvegarde pickle |
| `Frequence/src/predict.py` | Charge le pickle et génère les probabilités de sinistre |
| `Severite/src/train_rf_poids.py` | Entraîne le RF d'imputation des poids manquants |
| `Severite/src/main.py` | Orchestre entraînement et prédiction sévérité via `--mode` |
| `Severite/src/preprocessing.py` | Nettoyage, imputation poids, feature engineering véhicule |
| `Severite/src/train.py` | Entraîne CatBoostRegressor, sauvegarde pickle |
| `Severite/src/predict.py` | Charge le pickle et génère les montants de sinistre |
| `Prime/src/main.py` | Combine les deux modèles et calcule la prime pure |

---

## Modèles

### Modèle de fréquence

Prédit la probabilité qu'un assuré ait au moins un sinistre sur la période de couverture.

- **Algorithme** : CatBoostClassifier
- **Calibration** : Isotonic regression sur un set dédié (distinct du set de validation)
- **Métrique principale** : Calibration (moyenne des probas ≈ fréquence réelle observée)
- **AUC** : ~0.63

> **Pourquoi la calibration est la métrique principale ?**
> L'objectif n'est pas de classifier mais de produire des probabilités fiables pour calculer une prime juste. Une prime trop élevée fait perdre des clients, une prime trop basse fait perdre de l'argent.

**Split des données :**

| Set | Proportion | Rôle |
|---|---|---|
| Train | 60% | Entraînement du modèle |
| Validation | 20% | Early stopping |
| Calibration | 10% | Calibration isotonique |
| Test | 10% | Évaluation finale |

### Modèle de sévérité

Prédit le montant du sinistre lorsque celui-ci survient. Entraîné uniquement sur les contrats ayant eu au moins un sinistre.

- **Algorithme** : CatBoostRegressor
- **Métrique** : RMSE
- **Imputation** : Random Forest pour les poids véhicules manquants
- **Feature engineering** : ratios et interactions physiques du véhicule

---

## Feature engineering

### Fréquence
- Traitement des valeurs manquantes (`sex_conducteur2` → `"Aucun"`, `anciennete_vehicule` → médiane)
- Suppression des variables colinéaires (r > 0.95) et à importance nulle
- Discrétisation des âges en tranches métier : 18-30, 31-50, 51-70, 71+
- Extraction du département depuis le code postal (gestion Corse : 2A/2B)

### Sévérité
- Correction des anomalies manuelles (poids véhicule, cylindrée, prix)
- Correction des incohérences sur l'ancienneté du permis
- Imputation hiérarchique des poids manquants (groupby → Random Forest)
- Variables dérivées : rapport puissance/poids, énergie cinétique, sportivité du véhicule, ratios prix
