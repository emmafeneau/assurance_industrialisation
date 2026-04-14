"""
preprocessing.py — Nettoyage et feature engineering pour le modèle de sévérité.

Ce module regroupe toutes les étapes de transformation des données brutes :
1. Sélection des colonnes utiles
2. Correction des anomalies (poids, cylindrée, prix)
3. Correction des incohérences sur l'ancienneté du permis
4. Imputation des poids manquants (groupby hiérarchique + Random Forest)
5. Enrichissement par feature engineering (ratios, interactions physiques)

Note : le modèle Random Forest d'imputation des poids doit être entraîné
séparément et son pickle disponible dans models/model_rf_poids.pkl.
"""

import os

import joblib
import numpy as np
import pandas as pd

# -----------------------
# Colonnes
# -----------------------
COLS_TO_KEEP = [
    "index",
    "bonus",
    "type_contrat",
    "duree_contrat",
    "anciennete_info",
    "freq_paiement",
    "paiement",
    "utilisation",
    "code_postal",
    "conducteur2",
    "age_conducteur1",
    "age_conducteur2",
    "sex_conducteur1",
    "sex_conducteur2",
    "anciennete_permis1",
    "anciennete_permis2",
    "anciennete_vehicule",
    "cylindre_vehicule",
    "din_vehicule",
    "essence_vehicule",
    "marque_vehicule",
    "modele_vehicule",
    "debut_vente_vehicule",
    "fin_vente_vehicule",
    "vitesse_vehicule",
    "type_vehicule",
    "prix_vehicule",
    "poids_vehicule",
]

NUMERIC_COLS = [
    "din_vehicule",
    "poids_vehicule",
    "cylindre_vehicule",
    "vitesse_vehicule",
    "prix_vehicule",
]

SEVERITE_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RF_POIDS_PATH = os.path.normpath(
    os.path.join(SEVERITE_SRC_DIR, "..", "models", "model_rf_poids.pkl")
)

RF_POIDS_PATH = os.getenv("RF_POIDS_MODEL_PATH", DEFAULT_RF_POIDS_PATH)

RF_INPUT_COLS = ["essence_vehicule", "cylindre_vehicule", "din_vehicule", "vitesse_vehicule"]


# -----------------------
# Fonctions de nettoyage
# -----------------------


def select_columns(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
    """Sélectionne les colonnes utiles selon qu'on est en train ou test."""
    cols = COLS_TO_KEEP.copy()
    if target_cols:
        cols += [c for c in target_cols if c in df.columns]
    return df[[c for c in cols if c in df.columns]].copy()


def fix_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes numériques et remplace les infinis par NaN."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def fix_permis_anciennete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les incohérences sur l'ancienneté du permis :
    un conducteur ne peut pas avoir son permis avant ses 18 ans.
    """
    df["anciennete_permis1"] = np.where(
        (df["age_conducteur1"] - df["anciennete_permis1"]) >= 18,
        df["anciennete_permis1"],
        df["age_conducteur1"] - 18,
    )
    mask = ((df["age_conducteur2"] - df["anciennete_permis2"]) <= 18) & (df["conducteur2"] == "Yes")
    df.loc[mask, "anciennete_permis2"] = df.loc[mask, "age_conducteur2"] - 18
    return df


def fix_manual_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrections manuelles d'anomalies identifiées lors de l'exploration :
    - Véhicules avec un poids aberrant
    - Cylindrées à 0
    - Prix aberrants
    """
    # Poids aberrants
    if "index" in df.columns:
        df.loc[df["index"] == 71871, "poids_vehicule"] = 850
        df.loc[df["index"] == 93011, "poids_vehicule"] = 500
        # Cylindrées à 0
        df.loc[df["index"].isin([41748, 46971]), "cylindre_vehicule"] = 1199
        # Prix aberrants
        df.loc[df["index"] == 63424, "prix_vehicule"] = 14000
        df.loc[df["index"] == 71506, "prix_vehicule"] = 22497

    df.loc[df["cylindre_vehicule"] == 0, "cylindre_vehicule"] = 1
    return df


def impute_poids_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation hiérarchique des poids manquants par groupby.
    On descend progressivement dans la granularité jusqu'à trouver
    un groupe avec assez d'occurrences et un écart-type faible.
    """
    group_keys = [
        [
            "marque_vehicule",
            "modele_vehicule",
            "essence_vehicule",
            "cylindre_vehicule",
            "din_vehicule",
            "vitesse_vehicule",
            "type_vehicule",
        ],
        [
            "marque_vehicule",
            "modele_vehicule",
            "essence_vehicule",
            "cylindre_vehicule",
            "din_vehicule",
            "vitesse_vehicule",
        ],
        [
            "marque_vehicule",
            "modele_vehicule",
            "essence_vehicule",
            "cylindre_vehicule",
            "din_vehicule",
        ],
        ["marque_vehicule", "modele_vehicule", "essence_vehicule", "cylindre_vehicule"],
        ["marque_vehicule", "modele_vehicule", "essence_vehicule"],
        ["marque_vehicule", "modele_vehicule"],
        ["marque_vehicule"],
    ]

    stats_cols = ["occurrences", "poids_moyen", "poids_ecart_type"]

    for key in group_keys:
        df = df.drop(columns=[c for c in stats_cols if c in df.columns])

        df_stats = (
            df[df["poids_vehicule"] > 0]
            .groupby(key)
            .agg(
                occurrences=("poids_vehicule", "size"),
                poids_moyen=("poids_vehicule", "mean"),
                poids_ecart_type=("poids_vehicule", "std"),
            )
            .reset_index()
        )

        df = df.merge(df_stats, how="left", on=key)
        df["poids_moyen"] = df["poids_moyen"].fillna(0)

        mask = (
            (df["poids_vehicule"] == 0) & (df["occurrences"] > 10) & (df["poids_ecart_type"] < 100)
        )
        df.loc[mask, "poids_vehicule"] = df.loc[mask, "poids_moyen"].round().astype(int)

    df = df.drop(columns=[c for c in stats_cols if c in df.columns])
    return df


def impute_poids_by_rf(df: pd.DataFrame, rf_path: str = RF_POIDS_PATH) -> pd.DataFrame:
    """
    Imputation des poids restants (après groupby) via un modèle Random Forest.
    Le modèle doit être entraîné et disponible en pickle.
    Les véhicules Hybrid sont ignorés (hors du périmètre d'entraînement du RF).
    """
    if not os.path.exists(rf_path):
        print(f" Modèle RF poids introuvable ({rf_path}). Imputation RF ignorée.")
        return df

    rf = joblib.load(rf_path)
    mask_to_predict = df["poids_vehicule"] == 0
    mask_known = mask_to_predict & (df["essence_vehicule"] != "Hybrid")

    if mask_known.sum() == 0:
        return df

    df_to_predict = df.loc[mask_known, RF_INPUT_COLS].copy()
    df_to_predict["essence_vehicule"] = df_to_predict["essence_vehicule"].map(
        {"Diesel": 0, "Gasoline": 1}
    )
    df_to_predict = df_to_predict.dropna()

    predicted_weights = np.round(rf.predict(df_to_predict)).astype(int)
    df.loc[df_to_predict.index, "poids_vehicule"] = predicted_weights

    return df


def add_departement(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait le code département depuis le code postal."""

    def extract_dep(cp):
        cp = str(cp).zfill(5)
        if cp.startswith("20"):
            return "2A" if int(cp) <= 20199 else "2B"
        return cp[:2]

    df["code_departement"] = df["code_postal"].apply(extract_dep)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering : ratios et interactions physiques du véhicule.
    Ces variables capturent la sportivité, la dangerosité et la valeur du véhicule.
    """
    df["rapport_din_poids_vehicule"] = df["din_vehicule"] / df["poids_vehicule"]
    df["rapport_cylindre_poids_vehicule"] = df["cylindre_vehicule"] / df["poids_vehicule"]
    df["rapport_vitess_poids_vehicule"] = df["vitesse_vehicule"] / df["poids_vehicule"]
    df["energie_cinetique_vehicule"] = 0.5 * df["poids_vehicule"] * df["vitesse_vehicule"]
    df["produit_din_vitesse_vehicule"] = df["din_vehicule"] * df["vitesse_vehicule"]
    df["produit_din_cylindre_vehicule"] = df["din_vehicule"] * df["cylindre_vehicule"]
    df["carre_poids_vehicule"] = df["poids_vehicule"] ** 2
    df["carre_din_vehicule"] = df["din_vehicule"] ** 2
    df["carre_vitesse_vehicule"] = df["vitesse_vehicule"] ** 2
    df["rapport_din_cylindre_vehicule"] = df["din_vehicule"] / df["cylindre_vehicule"]
    df["sportivite_vehicule"] = (df["din_vehicule"] / df["poids_vehicule"]) * df["vitesse_vehicule"]
    df["rapport_din_prix_vehicule"] = df["din_vehicule"] / df["prix_vehicule"]
    df["rapport_cylindre_prix_vehicule"] = df["cylindre_vehicule"] / df["prix_vehicule"]
    df["rapport_vitesse_prix_vehicule"] = df["vitesse_vehicule"] / df["prix_vehicule"]
    df["rapport_poids_prix_vehicule"] = df["poids_vehicule"] / df["prix_vehicule"]
    return df


def get_feature_columns() -> tuple:
    """
    Retourne les listes de colonnes features (quanti et quali)
    utilisées pour entraîner le modèle CatBoost.
    """
    quanti = [
        "bonus",
        "duree_contrat",
        "anciennete_info",
        "age_conducteur1",
        "age_conducteur2",
        "anciennete_permis1",
        "anciennete_permis2",
        "anciennete_vehicule",
        "cylindre_vehicule",
        "din_vehicule",
        "debut_vente_vehicule",
        "fin_vente_vehicule",
        "vitesse_vehicule",
        "prix_vehicule",
        "poids_vehicule",
        "rapport_din_poids_vehicule",
        "rapport_cylindre_poids_vehicule",
        "rapport_vitess_poids_vehicule",
        "energie_cinetique_vehicule",
        "produit_din_vitesse_vehicule",
        "produit_din_cylindre_vehicule",
        "carre_poids_vehicule",
        "carre_din_vehicule",
        "carre_vitesse_vehicule",
        "rapport_din_cylindre_vehicule",
        "sportivite_vehicule",
        "rapport_din_prix_vehicule",
        "rapport_cylindre_prix_vehicule",
        "rapport_vitesse_prix_vehicule",
        "rapport_poids_prix_vehicule",
    ]
    quali = [
        "type_contrat",
        "freq_paiement",
        "paiement",
        "utilisation",
        "code_postal",
        "code_departement",
        "conducteur2",
        "sex_conducteur1",
        "sex_conducteur2",
        "essence_vehicule",
        "marque_vehicule",
        "modele_vehicule",
        "type_vehicule",
    ]
    return quanti, quali


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Pipeline complet de preprocessing pour le modèle de sévérité.

    Paramètres :
        df       : DataFrame brut chargé depuis CSV
        is_train : si True, filtre uniquement les contrats avec sinistre (nombre_sinistres > 0)

    Retourne un DataFrame prêt pour l'entraînement ou la prédiction.
    """
    target_cols = ["nombre_sinistres", "montant_sinistre"] if is_train else []
    df = select_columns(df, target_cols)
    df = fix_numeric_types(df)
    df = fix_manual_anomalies(df)
    df = fix_permis_anciennete(df)
    df = impute_poids_by_group(df)
    df = impute_poids_by_rf(df)
    df = add_departement(df)
    df = add_engineered_features(df)

    if is_train:
        df = df[df["nombre_sinistres"] > 0].copy()

    return df
