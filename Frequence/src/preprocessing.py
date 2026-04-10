import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset depuis un fichier CSV."""
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gestion des valeurs manquantes :
    - sex_conducteur2 : remplacé par 'Aucun' (pas de conducteur secondaire)
    - anciennete_vehicule : imputation par la médiane (1 seule valeur manquante)
    """
    df["sex_conducteur2"] = df["sex_conducteur2"].fillna("Aucun")
    df["anciennete_vehicule"] = df["anciennete_vehicule"].fillna(df["anciennete_vehicule"].median())
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suppression des colonnes inutiles :
    - Identifiants (index, id_client, id_vehicule, id_contrat)
    - Variables fortement corrélées à anciennete_vehicule (r > 0.95)
    - Variables à importance nulle dans CatBoost (conducteur2, type_vehicule)
    """
    cols_to_drop = [
        "index",
        "id_client",
        "id_vehicule",
        "id_contrat",
        "debut_vente_vehicule",
        "fin_vente_vehicule",
        "conducteur2",
        "type_vehicule",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop)


def create_age_tranches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Création des tranches d'âge pour les conducteurs principal et secondaire.
    Robuste à l'absence des colonnes age_conducteur2 et conducteur2.
    """
    # Conducteur principal
    df["age_conducteur1_tranche"] = (
        pd.cut(
            df["age_conducteur1"],
            bins=[18, 30, 50, 70, 150],
            labels=["18-30", "31-50", "51-70", "71+"],
            right=True,
            include_lowest=True,
        )
        .astype(str)
        .replace("nan", "Inconnu")
    )

    df = df.drop(columns=["age_conducteur1"])

    # Conducteur secondaire
    if "age_conducteur2" in df.columns:
        age2_cut = (
            pd.cut(
                df["age_conducteur2"],
                bins=[18, 30, 50, 70, 150],
                labels=["18-30", "31-50", "51-70", "71+"],
                include_lowest=True,
                right=True,
            )
            .astype(str)
            .replace("nan", "Aucun")
        )

        if "conducteur2" in df.columns:
            df["age_conducteur2_tranche"] = np.where(df["conducteur2"] == "No", "Aucun", age2_cut)
        else:
            df["age_conducteur2_tranche"] = age2_cut

        df = df.drop(columns=["age_conducteur2"])
    else:
        df["age_conducteur2_tranche"] = "Aucun"

    return df


def extract_departement(cp: str) -> str:
    """Extrait le département depuis un code postal."""
    if cp.startswith("20"):
        return "2A" if int(cp) <= 20199 else "2B"
    return cp[:2]


def create_departement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extraction du département depuis le code postal.
    Suppression du code postal brut (trop granulaire).
    """
    df["code_postal"] = df["code_postal"].astype(str).str.strip().str.zfill(5)
    df["departement"] = df["code_postal"].apply(extract_departement)
    df = df.drop(columns=["code_postal"])
    return df


def harmonize_sex_conducteur2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonisation du sexe du conducteur secondaire :
    - Pas de conducteur 2 → 'Aucun'
    - Sinon on conserve le sexe renseigné
    """
    if "conducteur2" in df.columns:
        df["sex_conducteur2"] = np.where(df["conducteur2"] == "No", "Aucun", df["sex_conducteur2"])
    df["sex_conducteur2"] = df["sex_conducteur2"].fillna("Aucun")
    return df


def get_cat_cols(df: pd.DataFrame, exclude: list = None) -> list:
    """Retourne la liste des colonnes catégorielles présentes dans le dataframe."""
    exclude = exclude or []
    return [c for c in df.columns if df[c].dtype in ["object", "category"] and c not in exclude]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de preprocessing :
    1. Gestion des valeurs manquantes
    2. Création des tranches d'âge (avant drop car utilise conducteur2)
    3. Harmonisation du sexe conducteur 2 (avant drop car utilise conducteur2)
    4. Suppression des colonnes inutiles
    5. Extraction du département
    6. Forçage des colonnes catégorielles en str propre (évite les erreurs CatBoost)
    """
    df = df.copy()
    df = handle_missing_values(df)
    df = create_age_tranches(df)
    df = harmonize_sex_conducteur2(df)
    df = drop_useless_columns(df)
    df = create_departement(df)

    # Force toutes les colonnes object/category en str propre
    # Évite que CatBoost tente de convertir "Aucun" en float
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype(str).replace("nan", "Aucun")

    return df
