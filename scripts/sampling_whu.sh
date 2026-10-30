#!/usr/bin/env bash
# End-to-end MMDiff sampling: OPT -> SAR -> IR.
# Usage: bash scripts/sampling.sh --prompt "a ship near the coast"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
PROMPT="${PROMPT:-An aerial view of a city with many buildings.}"

# The sampling modules import config/register/utils directly.
export PYTHONPATH="${PROJECT_ROOT}/mmdiff${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

"${PYTHON}" -c "import config, register, utils" \
  || { echo "[MMDiff] Unable to import the mmdiff package. Activate the mmdiff environment first." >&2; exit 1; }

echo "Using prompt: ${PROMPT}"

echo "[MMDiff] Step 1/3: generating OPT image and extracting spatial features..."
"${PYTHON}" "${PROJECT_ROOT}/mmdiff/sampling_OPT.py" \
  --n "city" --prompt "${PROMPT}" \
  --model_path "./models/MMDiff-WHU/mmdiff-opt" \
  --resolution 512 \
  --device cuda:0 \
  --attn_save_mode true --resnet_save_mode true \
  --attn_replace_mode false --resnet_replace_mode false \
  --output_dir "result-whu" \
  "$@"

echo "[MMDiff] Step 2/3: generating SAR image..."
"${PYTHON}" "${PROJECT_ROOT}/mmdiff/sampling_SAR.py" \
  --n "city" --prompt "${PROMPT}" \
  --model_path "./models/MMDiff-WHU/mmdiff-opt" \
  --sar_lora_path "./models/MMDiff-WHU/mmdiff-sar/sd-sar-lora-city" \
  --resolution 512 \
  --device cuda:0 \
  --attn_save_mode false --resnet_save_mode false \
  --attn_replace_mode true --resnet_replace_mode false \
  --output_dir "result-whu" \
  "$@"

echo "[MMDiff] Step 3/3: generating IR image..."
"${PYTHON}" "${PROJECT_ROOT}/mmdiff/sampling_IR.py" \
  --n "city" --prompt "${PROMPT}" \
  --model_path "./models/MMDiff-WHU/mmdiff-opt" \
  --ir_lora_path "./models/MMDiff-WHU/mmdiff-ir/sd-ir-lora-city" \
  --resolution 512 \
  --device cuda:0 \
  --attn_save_mode false --resnet_save_mode false \
  --attn_replace_mode true --resnet_replace_mode false \
  --output_dir "result-whu" \
  "$@"

echo "[MMDiff] Done. Results are in ${PROJECT_ROOT}/result-whu"