# mlforeng/cli/train.py

import argparse
import json

from mlforeng.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model with MLforEng."
    )

    parser.add_argument(
        "--dataset",
        choices=["synthetic", "commscom_churn"],
        default="synthetic",
        help="Which dataset to train on.",
    )

    parser.add_argument(
        "--model-name",
        default="logreg",
        help="Model name (defined in mlforeng.models, e.g. logreg, rf).",
    )

    # Synthetic-only arguments (ignored for commscom_churn)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples (synthetic dataset only).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Number of features (synthetic dataset only).",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test split.",
    )

    parser.add_argument(
        "--save-model-name",
        default="default_model",
        help="Name of folder under artifacts/pretrained/ to store the trained model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = TrainConfig(
        dataset=args.dataset,
        model_name=args.model_name,
        n_samples=args.n_samples,
        n_features=args.n_features,
        test_size=args.test_size,
        save_model_name=args.save_model_name,
    )

    results = train(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
