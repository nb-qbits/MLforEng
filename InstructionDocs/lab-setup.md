# Lab Environment: Python & Version Requirements
# Supported Python versions

For this workshop, you must use Python 3.10 or 3.11. We do not support Python 3.13 for the local lab at this time.
The reason is the Hugging Face tokenizers library:
It uses a Rust bridge (PyO3). PyO3 (and therefore tokenizers) currently supports up to Python 3.12.
With Python 3.13, it tries to compile from source and fails.

# Red Hat OpenShift AI
On Red Hat OpenShift AI, the official notebook images already use Python 3.11, so you are safe there by default.

# Local setup (Mac / Linux)

0. Clone the repository

$ git clone <git repo url>
$ cd <repo dir>

1. Install Python 3.10 or 3.11

On macOS with Homebrew:

$brew install python@3.11
python3.11 --version   # should show Python 3.11.x

2. Create a fresh virtual environment for the workshop

From your cloned MLforEng repository:

cd /path/to/MLforEng

# Remove any old venv that used Python 3.13
rm -rf .venv

# Create a new venv with Python 3.11
python3.11 -m venv .venv

# Activate it
$ source .venv/bin/activate

$ pip install -r requirements.txt  || true
# Upgrade basic tooling
$ python -m pip install --upgrade pip setuptools wheel


3. Install the required packages

MLforEng dependencies (your main requirements.txt):

$pip install -r requirements.txt

$pip install "datasets>=2.18.0" "pyarrow>=12.0.0"

# LLM fine-tuning dependencies:

$pip install -r mlforeng/llm_finetune/requirements.txt

# Check if all dependencies are installed correctly

python - << 'PY'
from mlforeng.llm_finetune import create_dataset
print("Imported mlforeng.llm_finetune.create_dataset OK")
PY

4. Run JupyterLab
$ pip install jupyterlab

$ jupyter-lab

5. Validate your setup

Open the 00_env_check.ipynb notebook and run it to validate your setup.

6. Run Workshop Notebooks

Open the 01_end_to_end_basics.ipynb notebook and run it to validate your setup.

