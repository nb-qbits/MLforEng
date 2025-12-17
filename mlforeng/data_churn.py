# mlforeng/data_churn.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASETS_DIR, RANDOM_SEED


# ---------- Paths to our churn files ----------

CHURN_CSV_PATH = DATASETS_DIR / "telecom_customer_churn.csv"
DICT_CSV_PATH = DATASETS_DIR / "telecom_data_dictionary.csv"
ZIP_POP_CSV_PATH = DATASETS_DIR / "telecom_zipcode_population.csv"


# ---------- Simple containers ----------

@dataclass
class ChurnFrame:
    features: pd.DataFrame   # X
    target: pd.Series        # y (0 = stayed, 1 = churned)


@dataclass
class ChurnSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


# ---------- Loaders ----------

def load_churn_raw() -> pd.DataFrame:
    """
    Load the raw ChurnTel customer churn table.
    """
    df = pd.read_csv(CHURN_CSV_PATH)
    return df

def load_churn_dictionary() -> pd.DataFrame:
    """
    Load the data dictionary with column descriptions.
    Useful for EDA and documentation.
    """
    # The file is not UTF-8; use a more permissive encoding.
    return pd.read_csv(DICT_CSV_PATH, encoding="latin1")

def load_zipcode_population() -> pd.DataFrame:
    """
    Load zipcode-level population table.
    We'll use this later for feature enrichment.
    """
    return pd.read_csv(ZIP_POP_CSV_PATH)


# ---------- Basic preprocessing for modeling ----------

def make_churn_frame() -> ChurnFrame:
    """
    Create a cleaned (X, y) frame for binary churn modeling.

    - Keep only customers with status Stayed or Churned.
    - Map target: 0 = Stayed, 1 = Churned.
    - Drop target / leakage columns from features.
    """
    df = load_churn_raw().copy()

    # Filter to a binary setting: drop "Joined" (new customers)
    df = df[df["Customer Status"].isin(["Stayed", "Churned"])].copy()

    # Create binary label
    df["churn_label"] = (df["Customer Status"] == "Churned").astype(int)

    # Columns that should NOT be used as features (ID, target, post-churn info)
    drop_cols = [
        "Customer Status",
        "Churn Category",
        "Churn Reason",
        "Customer ID",
    ]

    # Features = everything except drop_cols + label
    feature_df = df.drop(columns=drop_cols)

    # Move label out as separate target
    target = feature_df.pop("churn_label")

    return ChurnFrame(features=feature_df, target=target)


def train_test_churn(
    test_size: float = 0.2,
    stratify: bool = True,
) -> ChurnSplits:
    """
    Split churn data into train/test sets.

    Returns pandas DataFrames/Series; we will later decide
    how to encode them (one-hot, pipelines, etc.).
    """
    churn = make_churn_frame()

    stratify_vec = churn.target if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        churn.features,
        churn.target,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=stratify_vec,
    )

    return ChurnSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
