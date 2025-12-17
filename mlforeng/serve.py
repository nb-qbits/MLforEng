# mlforeng/serve.py

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predict import load_trained_model, predict_array, predict_dataframe


# ---------- Config ----------

DEFAULT_MODEL_NAME = os.getenv("MLFORENG_MODEL_NAME", "cli_logreg_test")


# ---------- Request / Response schemas ----------

class NumericPredictRequest(BaseModel):
    # For synthetic / numeric-only models
    instances: List[List[float]]


class NumericPredictResponse(BaseModel):
    model_name: str
    dataset: str | None
    n_instances: int
    predictions: List[int]


class ChurnPredictRequest(BaseModel):
    # For CommsCom churn models: each record is a dict of feature_name -> value
    records: List[Dict[str, Any]]


class ChurnPredictResponse(BaseModel):
    model_name: str
    dataset: str | None
    n_instances: int
    predictions: List[int]


# ---------- FastAPI app ----------

app = FastAPI(title="MLforEng Inference API")


@lru_cache(maxsize=1)
def get_loaded_model():
    """Load and cache the trained model specified by MLFORENG_MODEL_NAME."""
    return load_trained_model(DEFAULT_MODEL_NAME)


@app.get("/health")
def health():
    loaded = get_loaded_model()
    return {
        "status": "ok",
        "model_name": str(loaded.path.name),
        "dataset": loaded.dataset,
    }


# ---------- Numeric / synthetic prediction endpoint ----------

@app.post("/predict", response_model=NumericPredictResponse)
def predict_numeric(req: NumericPredictRequest):
    """
    Predict for numeric-only models (e.g., synthetic dataset).

    Expects:
      {
        "instances": [[f1, f2, ...], [...], ...]
      }
    """
    loaded = get_loaded_model()

    if loaded.dataset not in (None, "synthetic"):
        raise HTTPException(
            status_code=400,
            detail=f"/predict endpoint only supports 'synthetic' models, "
                   f"but current model dataset is '{loaded.dataset}'. "
                   f"Use /predict_churn for CommsCom churn models.",
        )

    try:
        X = np.array(req.instances, dtype=float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="instances must be 2D")

    preds = predict_array(loaded, X)

    return NumericPredictResponse(
        model_name=str(loaded.path.name),
        dataset=loaded.dataset,
        n_instances=X.shape[0],
        predictions=[int(p) for p in preds],
    )


# ---------- CommsCom churn prediction endpoint ----------

@app.post("/predict_churn", response_model=ChurnPredictResponse)
def predict_churn(req: ChurnPredictRequest):
    """
    Predict churn for CommsCom models trained on the churn dataset.

    Expects:
      {
        "records": [
          {"Age": 45, "Gender": "Male", "Contract": "Month-to-Month", ...},
          {...}
        ]
      }
    """
    loaded = get_loaded_model()

    if loaded.dataset != "commscom_churn":
        raise HTTPException(
            status_code=400,
            detail=f"/predict_churn endpoint requires a model trained on "
                   f"'commscom_churn' dataset, but current model dataset "
                   f"is '{loaded.dataset}'.",
        )

    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    df = pd.DataFrame(req.records)

    try:
        preds = predict_dataframe(loaded, df)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error during prediction: {e}",
        )

    return ChurnPredictResponse(
        model_name=str(loaded.path.name),
        dataset=loaded.dataset,
        n_instances=df.shape[0],
        predictions=[int(p) for p in preds],
    )
