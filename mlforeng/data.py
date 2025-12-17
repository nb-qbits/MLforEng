# mlforeng/data.py

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_example_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    test_size: float = 0.2,
) -> DatasetSplits:
    """
    Generate a synthetic dataset for classification.

    We choose n_informative and n_redundant *based on* n_features so that:
    n_informative + n_redundant < n_features  (to keep sklearn happy).
    """
    # Make sure we always have a valid combination
    # - at least 2 informative features
    # - some redundant features but never so many that they exceed n_features
    n_informative = max(2, n_features // 2)
    n_redundant = max(0, min(5, n_features - n_informative - 1))
    n_repeated = 0  # we don't explicitly use repeated features here

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        random_state=RANDOM_SEED,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    return DatasetSplits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
