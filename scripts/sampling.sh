#!/usr/bin/env bash
# End-to-end MMDiff sampling: OPT -> SAR -> IR.
# Usage: bash scripts/sampling.sh --prompt "a ship near the coast"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
PROMPT="${PROMPT:-There is a ship in the blue water on the shore.}"

# The sampling modules import config/register/utils directly.
export PYTHONPATH="${PROJECT_ROOT}/mmdiff${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

"${PYTHON}" -c "import config, register, utils" \
  || { echo "[MMDiff] Unable to import the mmdiff package. Activate the mmdiff environment first." >&2; exit 1; }

echo "Using prompt: ${PROMPT}"

echo "[MMDiff] Step 1/3: generating OPT image and extracting spatial features..."
"${PYTHON}" "${PROJECT_ROOT}/mmdiff/sampling_OPT.py" --n "ship" --prompt "${PROMPT}" --device cuda:0 "$@"

echo "[MMDiff] Step 2/3: generating SAR image..."
"${PYTHON}" "${PROJECT_ROOT}/mmdiff/sampling_SAR.py" --n "ship" --prompt "${PROMPT}" --device cuda:0 "$@"

echo "[MMDiff] Step 3/3: generating IR image..."
"${PYTHON}" "${PROJECT_ROOT}/mmdiff/sampling_IR.py" --n "ship" --prompt "${PROMPT}" --device cuda:0 "$@"

echo "[MMDiff] Done. Results are in ${PROJECT_ROOT}/result/."
