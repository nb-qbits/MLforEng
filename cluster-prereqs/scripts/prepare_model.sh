#!/usr/bin/env bash
set -euo pipefail

########################################
# 0. AUTH CHECKS (NO SECRETS HERE)
########################################

if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "❌ HUGGINGFACE_HUB_TOKEN is not set."
  echo "   Run:  source hf_creds.sh   before running this script."
  exit 1
fi

# Optional but nice: check AWS only if TARGET=s3
if [ "${TARGET:-s3}" = "s3" ] && [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  echo "❌ AWS credentials not set (AWS_ACCESS_KEY_ID missing)."
  echo "   Run:  source aws_creds.sh   or export AWS_* vars manually."
  exit 1
fi

########################################
# 1. CONFIG YOU EDIT PER MODEL (NO SECRETS)
########################################

# MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B}"     # HF repo id

# MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B}"      # local/S3 folder name
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Llama-3.2-1B-Instruct}"

LOCAL_ROOT="${LOCAL_ROOT:-/Users/vgrover/Downloads/software/AIWorkshops/MLforEng/cluster-prereqs/scripts/models}"

S3_BUCKET="${S3_BUCKET:-ocpmodel}"
S3_PREFIX="${S3_PREFIX:-${MODEL_NAME}}"

TARGET="${TARGET:-s3}"   # "s3" or "local"

########################################
# 2. DERIVED PATHS
########################################

LOCAL_DIR="${LOCAL_ROOT}/${MODEL_NAME}"
S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}"

echo "=== Model preparation ==="
echo " HF model id    : ${MODEL_ID}"
echo " Model name     : ${MODEL_NAME}"
echo " Local dir      : ${LOCAL_DIR}"
echo " S3 target URI  : ${S3_URI}"
echo " Target mode    : ${TARGET}"
echo

mkdir -p "${LOCAL_DIR}"

########################################
# 3. DOWNLOAD FROM HUGGING FACE
########################################

echo ">> Step 1: Downloading from Hugging Face to local..."
huggingface-cli download "${MODEL_ID}" \
  --local-dir "${LOCAL_DIR}" \
  --local-dir-use-symlinks False \
  --token "${HUGGINGFACE_HUB_TOKEN}"

echo "✅ Download complete: ${LOCAL_DIR}"

########################################
# 4. OPTIONAL: SYNC TO S3
########################################

if [ "${TARGET}" = "s3" ]; then
  echo
  echo ">> Step 2: Syncing local model to S3..."
  aws s3 sync "${LOCAL_DIR}" "${S3_URI}"
  echo "✅ Synced to ${S3_URI}"
fi

echo
echo "All done."
echo "You can now reference this model in OpenShift as:"
echo "  s3://${S3_BUCKET}/${S3_PREFIX}"
