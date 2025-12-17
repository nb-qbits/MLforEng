# mlforeng/predict.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np
import pandas as pd
from joblib import load

from .config import PRETRAINED_DIR


from typing import Optional

@dataclass
class LoadedModel:
    model: Any
    meta: Dict[str, Any]
    path: Path

    @property
    def dataset(self) -> Optional[str]:
        """
        Try to infer the dataset name from metadata.

        Priority:
        1. meta["extra"]["dataset"]  (used by cli.train)
        2. meta["config"]["dataset"] (used by tuning notebook)
        """
        extra = self.meta.get("extra", {})
        if isinstance(extra, dict) and "dataset" in extra:
            return extra["dataset"]

        cfg = self.meta.get("config", {})
        if isinstance(cfg, dict) and "dataset" in cfg:
            return cfg["dataset"]

        return None



def load_trained_model(name: str) -> LoadedModel:
    """
    Load a trained model and its metadata from artifacts/pretrained/<name>/.

    Assumes:
      - model.joblib
      - meta.json  (optional, but recommended)
    """
    model_dir = PRETRAINED_DIR / name
    model_fp = model_dir / "model.joblib"
    meta_fp = model_dir / "meta.json"

    if not model_fp.exists():
        raise FileNotFoundError(f"Model file not found: {model_fp}")

    model = load(model_fp)

    meta: Dict[str, Any] = {}
    if meta_fp.exists():
        try:
            meta = json.loads(meta_fp.read_text())
        except Exception:
            meta = {}

    return LoadedModel(model=model, meta=meta, path=model_dir)


def predict_array(loaded: LoadedModel, X: np.ndarray) -> np.ndarray:
    """
    Run predictions on a 2D numpy array X using the loaded model.

    This is mainly for the synthetic dataset models where the model
    was trained directly on numpy arrays.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape {X.shape}")
    return loaded.model.predict(X)


def predict_dataframe(loaded: LoadedModel, df: pd.DataFrame) -> np.ndarray:
    """
    Run predictions on a pandas DataFrame using the loaded model.

    This is intended for tabular datasets like the CommsCom churn data,
    where the saved model is a sklearn Pipeline that expects a DataFrame
    with named columns (for ColumnTransformer).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    return loaded.model.predict(df)
