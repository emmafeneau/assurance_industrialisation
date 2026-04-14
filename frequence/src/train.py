import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from preprocessing import get_cat_cols, preprocess
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# -----------------------
# Chemins
# -----------------------
TRAIN_PATH = "../../data/train.csv"
MODEL_PATH = "../models/catboost_calibrated.pkl"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split en 4 sets :
    - train (60%)  : entraînement du modèle
    - val   (20%)  : early stopping (use_best_model)
    - calib (10%)  : calibration des probabilités
    - test  (10%)  : évaluation finale (jamais vu pendant l'entraînement)

    Pourquoi 4 sets ?
    Le set de validation est utilisé par use_best_model=True pour sélectionner
    le meilleur nombre d'itérations. Si on calibre sur ce même set, la calibration
    isotonic va légèrement over-fitter. Un set de calibration distinct garantit
    des probabilités non biaisées.
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_tmp2, y_val, y_tmp2 = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_tmp2, y_tmp2, test_size=0.50, stratify=y_tmp2, random_state=42
    )

    print(f"Train  : {len(X_train):,} lignes ({len(X_train) / len(X):.0%})")
    print(f"Val    : {len(X_val):,} lignes ({len(X_val) / len(X):.0%})")
    print(f"Calib  : {len(X_calib):,} lignes ({len(X_calib) / len(X):.0%})")
    print(f"Test   : {len(X_test):,} lignes ({len(X_test) / len(X):.0%})")
    print(
        f"Fréquence sinistres — train: {y_train.mean():.3%} | calib: {y_calib.mean():.3%} | test: {y_test.mean():.3%}"
    )

    return X_train, X_val, X_calib, X_test, y_train, y_val, y_calib, y_test


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, cat_cols: list
) -> CatBoostClassifier:
    """
    Entraînement du modèle CatBoost.
    Pas de class_weights : sur un dataset de 50k+ lignes, CatBoost avec Logloss
    apprend correctement les vraies proportions. Les class_weights gonflent les
    probabilités et rendent la prime trop élevée même après calibration.
    """
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=200,
    )
    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val), use_best_model=True)
    return model


def calibrate_model(
    model: CatBoostClassifier, X_calib: pd.DataFrame, y_calib: pd.Series
) -> CalibratedClassifierCV:
    """
    Calibration isotonique sur un set dédié (distinct du set de validation).
    Isotonic = non-paramétrique, très bon avec 5 000+ lignes de calibration.
    Sigmoid (Platt scaling) = à privilégier si peu de données de calibration.
    """
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_calib, y_calib)
    print("Calibration activée (isotonic) sur set dédié.")
    return calibrator


def evaluate_model(
    calibrator: CalibratedClassifierCV, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """
    Évaluation finale sur le set de test.
    La métrique principale est la calibration (moyenne probas ≈ fréquence réelle)
    car l'objectif est la tarification, pas la classification.
    """
    y_proba = calibrator.predict_proba(X_test)[:, 1]

    print("\n--- Sanity check calibration ---")
    print("AUC (test)                  :", round(roc_auc_score(y_test, y_proba), 4))
    print("Fréquence réelle (test)     :", round(float(np.mean(y_test)), 4))
    print("Moyenne des probas (test)   :", round(float(np.mean(y_proba)), 4))
    print(
        "Quantiles probas (test)     :", np.quantile(y_proba, [0.1, 0.25, 0.5, 0.75, 0.9]).round(4)
    )


def save_model(calibrator: CalibratedClassifierCV, path: str) -> None:
    """Sauvegarde le modèle calibré."""
    joblib.dump(calibrator, path)
    print(f"\nModèle sauvegardé : {path}")


if __name__ == "__main__":
    # 1. Chargement
    print("Chargement des données...")
    df = pd.read_csv(TRAIN_PATH)

    # 2. Preprocessing
    print("Preprocessing...")
    df = preprocess(df)

    # 3. Séparation features / cible
    LEAK_COLS = ["montant_sinistre"]
    X = df.drop(columns=["nombre_sinistres"] + LEAK_COLS)
    y = df["nombre_sinistres"]

    # 4. Colonnes catégorielles
    cat_cols = get_cat_cols(X)
    print("Colonnes catégorielles :", cat_cols)

    # 5. Split
    X_train, X_val, X_calib, X_test, y_train, y_val, y_calib, y_test = split_data(X, y)

    # 6. Entraînement
    print("\nEntraînement du modèle...")
    model = train_model(X_train, y_train, X_val, y_val, cat_cols)

    # 7. Calibration
    calibrator = calibrate_model(model, X_calib, y_calib)

    # 8. Évaluation
    evaluate_model(calibrator, X_test, y_test)

    # 9. Sauvegarde
    save_model(calibrator, MODEL_PATH)
