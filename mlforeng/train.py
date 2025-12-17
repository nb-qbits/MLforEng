# mlforeng/train.py

from dataclasses import dataclass, asdict
from typing import Dict, Any
from pathlib import Path
import json

from joblib import dump
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .data import load_example_dataset
from .data_churn import train_test_churn
from .models import create_local_model
from .config import PRETRAINED_DIR


@dataclass
class TrainConfig:
    # What dataset to train on
    # - "synthetic": the toy numeric dataset (Module 01)
    # - "commscom_churn": the real CommsCom churn dataset
    dataset: str = "synthetic"

    # Model choice (resolved via models.py)
    model_name: str = "logreg"

    # Synthetic dataset only:
    n_samples: int = 1000
    n_features: int = 20

    # Common:
    test_size: float = 0.2
    save_model_name: str = "default_model"  # folder under artifacts/pretrained


def _train_synthetic(config: TrainConfig) -> Dict[str, Any]:
    """
    Original toy training path on synthetic numeric data.
    Uses create_local_model directly on numpy arrays.
    """
    splits = load_example_dataset(
        n_samples=config.n_samples,
        n_features=config.n_features,
        test_size=config.test_size,
    )

    model = create_local_model(config.model_name)
    model.fit(splits.X_train, splits.y_train)

    y_pred = model.predict(splits.X_test)
    acc = accuracy_score(splits.y_test, y_pred)

    metrics = {"accuracy": float(acc)}

    return {
        "model": model,
        "metrics": metrics,
        "extra": {
            "dataset": "synthetic",
            "n_samples": config.n_samples,
            "n_features": config.n_features,
        },
    }


def _train_commscom_churn(config: TrainConfig) -> Dict[str, Any]:
    """
    Training path for the real CommsCom customer churn dataset.

    Steps:
      - load train/test splits (pandas DataFrames)
      - build preprocessing pipeline:
          - impute + scale numeric features
          - impute + one-hot encode categorical features
      - attach the base model (logreg, rf, etc.)
      - train and compute accuracy + ROC-AUC (when possible)
    """
    splits = train_test_churn(test_size=config.test_size, stratify=True)

    X_train = splits.X_train
    X_test = splits.X_test
    y_train = splits.y_train
    y_test = splits.y_test

    # Identify column types
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Pipelines for numeric and categorical features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # Base model from our registry (logreg, rf, etc.)
    base_model = create_local_model(config.model_name)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    metrics: Dict[str, Any] = {"accuracy": float(acc)}

    # Try to compute ROC-AUC if the model supports predict_proba
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            metrics["roc_auc"] = float(roc_auc)
        except Exception:
            pass

    return {
        "model": model,
        "metrics": metrics,
        "extra": {
            "dataset": "commscom_churn",
            "n_train_rows": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        },
    }


def train(config: TrainConfig) -> Dict[str, Any]:
    """
    High-level training entrypoint.

    - Selects dataset ("synthetic" vs "commscom_churn")
    - Trains the chosen model
    - Saves model + metadata to artifacts/pretrained/<save_model_name>/
    - Returns a dict with model_path, config, metrics, etc.
    """
    if config.dataset == "synthetic":
        result = _train_synthetic(config)
    elif config.dataset == "commscom_churn":
        result = _train_commscom_churn(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    model = result["model"]
    metrics = result["metrics"]
    extra = result.get("extra", {})

    # Prepare directory for saving
    model_dir: Path = PRETRAINED_DIR / config.save_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_fp = model_dir / "model.joblib"
    meta_fp = model_dir / "meta.json"

    dump(model, model_fp)

    meta = {
        "config": asdict(config),
        "metrics": metrics,
        "extra": extra,
    }
    meta_fp.write_text(json.dumps(meta, indent=2))

    # Return a summary
    return {
        "model_path": str(model_fp),
        "config": meta["config"],
        "metrics": metrics,
        "extra": extra,
    }
