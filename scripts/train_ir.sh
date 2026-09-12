#!/usr/bin/env bash
# Train the IR LoRA branch with the paper's training configuration.
set -euo pipefail

project_root="path/to/MMDiff"
cd "$project_root"
python_bin="${PYTHON:-python}"
category="${CATEGORY:-ship}"

exec "$python_bin" -m accelerate.commands.launch --num_processes=2 --mixed_precision=fp16 \
    "$project_root/mmdiff/train.py" \
    --pretrained_model_name_or_path="$project_root/model-v1/mmdiff-opt" \
    --dataset_modality=ir \
    --dataset_category="$category" \
    --output_dir="$project_root/model-v1/mmdiff-ir/sd-ir-lora-$category" \
    --resolution=256 \
    --train_batch_size=1 \
    --max_train_steps=15000 \
    --learning_rate=1e-4 \
    --checkpointing_steps=5000 \
    "$@"
