# pipelines/commscom_churn_pipeline_v2.py

from pathlib import Path
import json
import shutil

from kfp import dsl
from kfp.dsl import Input, Output, Artifact

TRAIN_IMAGE = "quay.io/vgrover/mlforeng-churn-train:amd64"


# 1. TRAIN STEP ----------------------------------------------------------------
@dsl.component(
    base_image=TRAIN_IMAGE,
)
def train_commscom_churn(
    # IMPORTANT: artifact comes FIRST, no default
    model_dir: Output[Artifact],
    model_family: str = "rf",
    test_size: float = 0.2,
    save_model_name: str = "commscom_rf_pipeline",
):
    """
    Step 1: Train the CommsCom churn model and write artifacts into model_dir.
    """
    import os
    from pathlib import Path
    import shutil

    # Configure MLforEng training via env vars
    os.environ["MLFORENG_MODEL_FAMILY"] = model_family
    os.environ["MLFORENG_TEST_SIZE"] = str(test_size)
    os.environ["MLFORENG_SAVE_MODEL_NAME"] = save_model_name
    os.environ["MLFORENG_OUTPUT_DIR"] = "/tmp/output"

    out_dir = Path("/tmp/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== CommsCom churn TRAIN step ===")
    print(f"Model family : {model_family}")
    print(f"Test size    : {test_size}")
    print(f"Save name    : {save_model_name}")
    print(f"Output dir   : {out_dir}")

    # Run the training step from our module (inside the image)
    from mlforeng.pipeline.train_churn_step import main as train_main

    train_main()

    # model_dir is an Artifact; .path is where KFP expects us to write files
    model_dir_path = Path(model_dir.path)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    for p in out_dir.iterdir():
        if p.is_file():
            print(f"Copying artifact {p.name} -> {model_dir_path}")
            shutil.copy2(p, model_dir_path / p.name)

    print("Contents of model_dir:", list(model_dir_path.iterdir()))
    print("=== TRAIN step complete ===")


# 2. EVALUATE STEP -------------------------------------------------------------
@dsl.component(
    base_image=TRAIN_IMAGE,
)
def evaluate_commscom_model(
    model_dir: Input[Artifact],
    metrics_dir: Output[Artifact],
):
    """
    Step 2: Evaluate the trained model.

    - Reads meta.json from model_dir.
    - Writes metrics.json into metrics_dir.
    """
    from pathlib import Path
    import json

    model_dir_path = Path(model_dir.path)
    meta_path = model_dir_path / "meta.json"

    print("=== CommsCom churn EVALUATE step ===")
    print("Model dir:", model_dir_path)
    print("Meta path:", meta_path)

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {model_dir_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    metrics = meta.get("metrics", {})
    print("Loaded metrics:", metrics)

    metrics_dir_path = Path(metrics_dir.path)
    metrics_dir_path.mkdir(parents=True, exist_ok=True)

    metrics_out_path = metrics_dir_path / "metrics.json"
    with metrics_out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Wrote metrics to:", metrics_out_path)
    print("=== EVALUATE step complete ===")


# 3. REGISTER STEP -------------------------------------------------------------
@dsl.component(
    base_image=TRAIN_IMAGE,
)
def register_commscom_model(
    model_dir: Input[Artifact],
):
    """
    Step 3: 'Register' the model (placeholder).

    For now, just checks that model.joblib exists and logs the path.
    """
    from pathlib import Path

    model_dir_path = Path(model_dir.path)
    print("=== CommsCom churn REGISTER step ===")
    print("Pretending to register model from:", model_dir_path)

    if not (model_dir_path / "model.joblib").exists():
        raise FileNotFoundError("model.joblib not found; cannot register.")

    print("Model artifact present. Registration placeholder complete.")
    print("=== REGISTER step complete ===")


# PIPELINE DEFINITION ----------------------------------------------------------
@dsl.pipeline(
    name="commscom-churn-training-pipeline-v2",
    description="Train, evaluate, and (placeholder) register CommsCom churn model with mlforeng on OpenShift AI.",
)
def commscom_churn_training_pipeline_v2(
    model_family: str = "rf",
    test_size: float = 0.2,
):
    # Step 1: train
    train_step = train_commscom_churn(
        model_family=model_family,
        test_size=test_size,
        save_model_name="commscom_rf_pipeline",
    )

    # Step 2: evaluate (depends on train)
    eval_step = evaluate_commscom_model(
        model_dir=train_step.outputs["model_dir"],
    )

    # Step 3: register (also depends on train)
    _ = register_commscom_model(
        model_dir=train_step.outputs["model_dir"],
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=commscom_churn_training_pipeline_v2,
        package_path="commscom_churn_pipeline_v2.yaml",
    )
