# mlforeng/cli/serve.py

import argparse
import os

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve MLforEng model via FastAPI.")
    parser.add_argument(
        "--model-name",
        default="cli_logreg_test",
        help="Trained model directory under artifacts/pretrained/ to load.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()

    # Tell mlforeng.serve which model to load
    os.environ["MLFORENG_MODEL_NAME"] = args.model_name

    uvicorn.run(
        "mlforeng.serve:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
