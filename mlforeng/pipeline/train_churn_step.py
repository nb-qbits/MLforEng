"""
Pipeline training step for CommsCom churn on OpenShift AI.

This script is designed to run inside a container as a single pipeline step.
It:
- loads the CommsCom churn dataset via mlforeng.data_churn,
- trains a model (RF by default),
- evaluates it,
- writes model + meta.json to an OUTPUT_DIR (local path in the container).

On OpenShift AI Data Science Pipelines, that OUTPUT_DIR will be captured as
an output artifact (backed by S3/MinIO via the pipeline server connection).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from joblib import dump
from sklearn.metrics import roc_auc_score, classification_report

from mlforeng.data_churn import train_test_churn
from mlforeng.models import create_local_model  # your existing factory
from mlforeng.train import TrainConfig

# --------------------------------------------------------------------
# Read configuration from environment variables (pipeline-friendly)
# --------------------------------------------------------------------

MODEL_FAMILY = os.getenv("MLFORENG_MODEL_FAMILY", "rf")  # "logreg" or "rf"
TEST_SIZE = float(os.getenv("MLFORENG_TEST_SIZE", "0.2"))
SAVE_MODEL_NAME = os.getenv("MLFORENG_SAVE_MODEL_NAME", "commscom_rf_pipeline")
OUTPUT_DIR = Path(os.getenv("MLFORENG_OUTPUT_DIR", "artifacts/pipeline_output"))

DATASET_NAME = "commscom_churn"


def main():
    print("=== CommsCom churn pipeline training step ===")
    print(f"Model family: {MODEL_FAMILY}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Save model name: {SAVE_MODEL_NAME}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load data (same split logic as your other modules)
    # ----------------------------------------------------------------
    splits = train_test_churn(test_size=TEST_SIZE, stratify=True)
    X_train, X_test = splits.X_train, splits.X_test
    y_train, y_test = splits.y_train, splits.y_test

    # ----------------------------------------------------------------
    # 2. Build model via your unified factory
    #    (we just let the pipeline handle preprocessing)
    # ----------------------------------------------------------------
    # For now we keep this simple and re-use the classical path:
    cfg = TrainConfig(
        dataset=DATASET_NAME,
        model_name=MODEL_FAMILY,
        test_size=TEST_SIZE,
        save_model_name=SAVE_MODEL_NAME,
    )

    # NOTE: we can either:
    # - call your existing `train(cfg)` function and then copy artifacts, OR
    # - inline training logic here.
    #
    # To keep it explicit for a pipeline step, we'll inline the core logic
    # similar to your other notebooks.

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    if MODEL_FAMILY == "logreg":
        base_model = LogisticRegression(max_iter=1000)
    else:
        base_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    # ----------------------------------------------------------------
    # 3. Train and evaluate
    # ----------------------------------------------------------------
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    print("=== Classification report (test) ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC–AUC:", roc_auc)

    # ----------------------------------------------------------------
    # 4. Save model + meta.json to OUTPUT_DIR
    # ----------------------------------------------------------------
    model_fp = OUTPUT_DIR / "model.joblib"
    dump(pipeline, model_fp)
    print(f"Saved model to {model_fp}")

    meta = {
        "config": {
            "model_name": MODEL_FAMILY,
            "dataset": DATASET_NAME,
            "save_model_name": SAVE_MODEL_NAME,
            "test_size": TEST_SIZE,
        },
        "metrics": {
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "n_train_rows": int(len(X_train)),
            "n_test_rows": int(len(X_test)),
        },
        "extra": {
            "dataset": DATASET_NAME,
            "source": "pipeline_step",
        },
    }

    meta_fp = OUTPUT_DIR / "meta.json"
    meta_fp.write_text(json.dumps(meta, indent=2))
    print(f"Saved meta to {meta_fp}")

    print("=== Training step complete ===")


if __name__ == "__main__":
    main()
