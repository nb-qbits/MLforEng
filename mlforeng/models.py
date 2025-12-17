# mlforeng/models.py

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelFamily(str, Enum):
    LOGREG = "logreg"
    RANDOM_FOREST = "random_forest"
    BERT = "bert"
    LLAMA_3 = "llama3"


@dataclass
class ModelSpec:
    """
    High-level description of a model.
    - family: what kind of stack it is
    - hf_model_id: Hugging Face model id (for transformers-based models)
    - task_type: e.g. 'classification', 'causal_lm'
    - extra: free-form dict for special flags (e.g. LoRA config)
    """
    family: ModelFamily
    hf_model_id: Optional[str] = None
    task_type: str = "classification"
    extra: Dict[str, Any] = None


def get_model_spec(name: str) -> ModelSpec:
    """
    Map a short, user-facing model name to a full spec.
    This is where you define *all* supported models.
    """
    # Small, local classical models
    if name == "logreg":
        return ModelSpec(family=ModelFamily.LOGREG)

    if name in ("rf", "random_forest"):
        return ModelSpec(family=ModelFamily.RANDOM_FOREST)

    # Transformer classifier, like Made-With-ML's SciBERT
    if name == "bert_scibert":
        return ModelSpec(
            family=ModelFamily.BERT,
            hf_model_id="allenai/scibert_scivocab_uncased",
            task_type="sequence_classification",
        )

    # Llama 3.1 8B Instruct – the one you’ll fine-tune on OpenShift AI
    # (exact id can be adjusted to whatever your OpenShift AI setup uses)
    if name == "llama3_8b_instruct":
        return ModelSpec(
            family=ModelFamily.LLAMA_3,
            hf_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            task_type="causal_lm",
            extra={"recommended_backend": "openshift_ai_ray"},
        )

    raise ValueError(f"Unknown model name: {name}")


def create_local_model(model_name: str):
    """
    Create a model object for LOCAL runs only (laptop, single container).
    Big Llama models should not be instantiated here – they are handled
    by distributed OpenShift AI workflows.
    """
    spec = get_model_spec(model_name)

    if spec.family == ModelFamily.LOGREG:
        return LogisticRegression(max_iter=1000)

    if spec.family == ModelFamily.RANDOM_FOREST:
        return RandomForestClassifier(n_estimators=100)

    if spec.family == ModelFamily.BERT:
        # basic example; in practice you'd wrap this in a nicer class
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(
            spec.hf_model_id,
            num_labels=2,   # adapt as needed
        )

    if spec.family == ModelFamily.LLAMA_3:
        # Intentionally blocked here: Llama 3.x is too big for the "simple local trainer".
        # Instead, your OpenShift AI module will use spec.hf_model_id and submit a
        # distributed fine-tuning job (Ray / Kubeflow Trainer).
        raise RuntimeError(
            "Llama 3.x models are handled by the OpenShift AI fine-tuning module, "
            "not by the local trainer. Use the Llama3 workshop module instead."
        )

    raise ValueError(f"Unhandled model family: {spec.family}")

def create_model(name: str):
    """
    Backwards-compatible alias so older code that imports create_model
    still works. Internally we use create_local_model.
    """
    return create_local_model(name)