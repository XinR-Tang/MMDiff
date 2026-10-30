#!/usr/bin/env bash
# Train the IR LoRA branch on mmdiff-whu, one category at a time, 5 epochs each.
set -euo pipefail

project_root="Path/to/MMDiff"
data_root="Path/to/MMDiff-WHU"
cd "$project_root"

# Override the executable if needed, e.g. PYTHON=/path/to/env/bin/python.
python_bin="${PYTHON:-python}"

epochs=5
num_processes=2
batch_size=1
grad_accum=1
# Override the loop with, e.g. CATEGORIES="city water".
categories="${CATEGORIES:-city farmland forest water}"

for category in $categories; do
    # Count this category's images and convert epochs to optimization steps,
    # mirroring train.py: per_epoch = ceil(ceil(ceil(N/b)/P)/G)
    n_images=$(find "$data_root/sar/train/$category" -name '*.jpg' | wc -l | tr -d ' ')
    micro_batches=$(( (n_images + batch_size - 1) / batch_size ))
    after_shard=$(( (micro_batches + num_processes - 1) / num_processes ))
    per_epoch=$(( (after_shard + grad_accum - 1) / grad_accum ))
    max_steps=$(( epochs * per_epoch ))

    echo "sar/${category}: ${n_images} images, ${per_epoch} steps/epoch, ${max_steps} steps (${epochs} epochs)"

    "$python_bin" -m accelerate.commands.launch --num_processes=$num_processes --mixed_precision=fp16 \
        "$project_root/mmdiff/train.py" \
        --pretrained_model_name_or_path="$project_root/models/MMDiff-WHU/mmdiff-opt" \
        --dataset_modality=sar \
        --dataset_category="$category" \
        --is_OSI=False \
        --optical_sar_infrared_data_dir="$data_root" \
        --r 4 \
        --lora_alpha 4 \
        --output_dir="$project_root/models/MMDiff-WHU/mmdiff-sar/sd-sar-lora-$category" \
        --resolution=512 \
        --train_batch_size=$batch_size \
        --gradient_accumulation_steps=$grad_accum \
        --num_train_epochs=$epochs \
        --max_train_steps=$max_steps \
        --learning_rate=1e-4 \
        --checkpointing_steps=$per_epoch \
        "$@"
done
