# mlforeng/cli/predict.py

import argparse
import json
import numpy as np

from mlforeng.predict import load_trained_model, predict_array
from mlforeng.data import load_example_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictions with a trained MLforEng model."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name of the trained model directory under artifacts/pretrained/",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="How many samples from the test set to predict on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    loaded = load_trained_model(args.model_name)

    # For now, just use freshly generated test data with the same feature dim.
    # (Later, you'll load a real test set.)
    meta_cfg = loaded.meta.get("config", {})
    n_features = meta_cfg.get("n_features", 10)

    # Generate a dataset with same feature count and take test split
    splits = load_example_dataset(
        n_samples=200, n_features=n_features, test_size=0.2
    )
    X_test = splits.X_test[: args.num_samples]

    preds = predict_array(loaded, X_test)

    output = {
        "model_name": args.model_name,
        "meta": loaded.meta,
        "n_samples": int(X_test.shape[0]),
        "predictions": preds.tolist(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
