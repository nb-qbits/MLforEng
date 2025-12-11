# mlforeng/config.py

from pathlib import Path
import os

# Root of the project; overridable via env for containers/K8s/OpenShift AI
PROJECT_ROOT = Path(
    os.getenv("MLFORENG_ROOT", Path(__file__).resolve().parents[1])
)

# Base artifacts directory (models, datasets, etc.)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Datasets directory (CSV, parquet, etc.)
DATASETS_DIR = ARTIFACTS_DIR / "datasets"

# Backwards-compatible alias (if older code uses DATA_DIR)
DATA_DIR = DATASETS_DIR

# Where trained models / artifacts are stored
PRETRAINED_DIR = ARTIFACTS_DIR / "pretrained"

# Global random seed (overridable via env)
RANDOM_SEED = int(os.getenv("MLFORENG_RANDOM_SEED", "42"))
