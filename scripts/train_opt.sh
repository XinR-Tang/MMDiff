#!/usr/bin/env bash
# Train the OPT branch with the paper's training configuration.
set -euo pipefail

project_root="path/to/MMDiff"
cd "$project_root"

# Override the executable if needed, e.g. PYTHON=/path/to/env/bin/python.
python_bin="${PYTHON:-python}"

exec "$python_bin" -m accelerate.commands.launch --num_processes=2 --mixed_precision=fp16 \
    "$project_root/mmdiff/train.py" \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
    --dataset_modality=opt \
    --output_dir="$project_root/model-v1/mmdiff-opt" \
    --resolution=256 \
    --train_batch_size=1 \
    --max_train_steps=15000 \
    --learning_rate=1e-4 \
    --checkpointing_steps=5000 \
    "$@"
