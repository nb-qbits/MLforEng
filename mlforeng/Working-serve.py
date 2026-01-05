from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import shap
from functools import lru_cache
from typing import List, Dict, Any
import requests

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predict import load_trained_model, predict_array, predict_dataframe

RHOAI_LLM_BASE_URL = os.getenv("RHOAI_LLM_BASE_URL")
RHOAI_LLM_API_KEY = os.getenv("RHOAI_LLM_API_KEY")
RHOAI_LLM_MODEL = os.getenv("RHOAI_LLM_MODEL")
# ---------- Config ----------

DEFAULT_MODEL_NAME = os.getenv("MLFORENG_MODEL_NAME", "commscom_rf_tuned")


# ---------- Request / Response schemas ----------

class NumericPredictRequest(BaseModel):
    instances: List[List[float]]

class ChurnExplainResponse(BaseModel):
    model_name: str
    dataset: str | None
    top_features: List[Dict[str, Any]]

class NumericPredictResponse(BaseModel):
    model_name: str
    dataset: str | None
    n_instances: int
    predictions: List[int]


class ChurnPredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class ChurnPredictResponse(BaseModel):
    model_name: str
    dataset: str | None
    n_instances: int
    predictions: List[int]
    probabilities: List[float | None]

class ChurnLLMSummaryRequest(BaseModel):
    customer_id: str
    prediction: int
    probability: float | None
    top_features: List[Dict[str, Any]]

class ChurnLLMSummaryResponse(BaseModel):
    summary: str


# ---------- FastAPI app ----------

app = FastAPI(title="MLforEng Inference API")

print(">>> LOADED serve.py FROM:", __file__)


# ---------- Model loader ----------

@lru_cache(maxsize=1)
def get_loaded_model():
    loaded = load_trained_model(DEFAULT_MODEL_NAME)

    # dataset is derived from meta
    if loaded.meta is None:
        loaded.meta = {}

    loaded.meta["dataset"] = "commscom_churn"
    return loaded


# ---------- Health ----------

@app.get("/health")
def health():
    loaded = get_loaded_model()
    return {
        "status": "ok",
        "model_name": str(loaded.path.name),
        "dataset": loaded.dataset,
    }

def llm_summarize(prompt: str) -> str:
    if not RHOAI_LLM_BASE_URL or not RHOAI_LLM_API_KEY:
        raise RuntimeError("LLM environment variables not set")

    url = f"{RHOAI_LLM_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {RHOAI_LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": RHOAI_LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior telecom support engineer assistant. "
                    "Summarize churn risk clearly for human operators. "
                    "Do NOT hallucinate. Base your answer only on provided facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)

    # 🔥 CRITICAL: surface real error
    if resp.status_code != 200:
        raise RuntimeError(
            f"LLM error {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

@app.post("/llm/summarize_churn")
def summarize_churn(req: ChurnLLMSummaryRequest):
    try:
        lines = [
            f"Prediction: {'Churn' if req.prediction == 1 else 'No churn'}",
            f"Probability: {req.probability:.2%}" if req.probability is not None else "Probability: N/A",
            "Top factors:",
        ]

        for f in req.top_features:
            lines.append(f"- {f['feature']} ({f['direction']})")

        prompt = "\n".join(lines)

        summary = llm_summarize(prompt)

        return {"summary": summary}

    except Exception as e:
        # 👇 THIS IS WHAT YOU WERE MISSING
        raise HTTPException(
            status_code=500,
            detail=f"LLM summarize failed: {str(e)}",
        )




# ---------- Numeric / synthetic ----------

@app.post("/predict", response_model=NumericPredictResponse)
def predict_numeric(req: NumericPredictRequest):
    loaded = get_loaded_model()

    if loaded.dataset not in (None, "synthetic"):
        raise HTTPException(
            status_code=400,
            detail=f"/predict supports only synthetic models, "
                   f"but current dataset is '{loaded.dataset}'.",
        )

    X = np.array(req.instances, dtype=float)

    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="instances must be 2D")

    preds = predict_array(loaded, X)

    return NumericPredictResponse(
        model_name=str(loaded.path.name),
        dataset=loaded.dataset,
        n_instances=X.shape[0],
        predictions=[int(p) for p in preds],
    )

@app.post("/explain_churn")
def explain_churn(req: ChurnPredictRequest):
    loaded = get_loaded_model()

    if loaded.dataset != "commscom_churn":
        raise HTTPException(
            status_code=400,
            detail="Explanation requires commscom_churn model",
        )

    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    df = pd.DataFrame(req.records)
    pipeline = loaded.model

    # ---- extract estimator ----
    try:
        estimator = pipeline[-1]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Model is not a sklearn Pipeline",
        )

    # ---- get feature names ----
    try:
        feature_names = pipeline[:-1].get_feature_names_out()
    except Exception:
        feature_names = df.columns.tolist()

    # ---- get importance / coefficients ----
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = estimator.coef_[0]
    else:
        raise HTTPException(
            status_code=500,
            detail="Model does not support explainability",
        )

    contributions = sorted(
        zip(feature_names, importances),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    top_features = [
        {
            "feature": name,
            "impact": float(val),
            "direction": "increases risk" if val > 0 else "reduces risk",
        }
        for name, val in contributions
    ]

    return {
        "model_name": str(loaded.path.name),
        "dataset": loaded.dataset,
        "top_features": top_features,
    }



# ---------- CommsCom churn ----------

@app.post("/predict_churn", response_model=ChurnPredictResponse)
def predict_churn(req: ChurnPredictRequest):
    loaded = get_loaded_model()

    if loaded.dataset != "commscom_churn":
        raise HTTPException(
            status_code=400,
            detail=f"/predict_churn requires 'commscom_churn' model, "
                   f"but current dataset is '{loaded.dataset}'.",
        )

    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    df = pd.DataFrame(req.records)

    # ---- predictions (pipeline-safe) ----
    preds = predict_dataframe(loaded, df)

    # ---- probabilities (pipeline-safe) ----
    try:
        probs = loaded.model.predict_proba(df)[:, 1]
    except Exception:
        probs = [None] * len(preds)

    return ChurnPredictResponse(
        model_name=str(loaded.path.name),
        dataset=loaded.dataset,
        n_instances=len(preds),
        predictions=[int(p) for p in preds],
        probabilities=[float(p) if p is not None else None for p in probs],
    )
