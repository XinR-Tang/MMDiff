#!/usr/bin/env bash
# Train the OPT branch on mmdiff-whu (all categories) for 5 epochs.
set -euo pipefail

project_root="Path/to/MMDiff"
data_root="Path/to/MMDiff-WHU"
cd "$project_root"

# Override the executable if needed, e.g. PYTHON=/path/to/env/bin/python.
python_bin="${PYTHON:-python}"

epochs=5
num_processes=2 # GPU memeory
batch_size=1
grad_accum=1

# Count training images (all categories) and convert epochs to optimization
# steps, mirroring train.py:
#   per_epoch = ceil(ceil(ceil(N/b)/P)/G)
n_images=$(find "$data_root/opt/train" -name '*.jpg' | wc -l | tr -d ' ')
micro_batches=$(( (n_images + batch_size - 1) / batch_size ))
after_shard=$(( (micro_batches + num_processes - 1) / num_processes ))
per_epoch=$(( (after_shard + grad_accum - 1) / grad_accum ))
max_steps=$(( epochs * per_epoch ))

echo "opt: ${n_images} images, ${per_epoch} steps/epoch, ${max_steps} steps (${epochs} epochs)"

exec "$python_bin" -m accelerate.commands.launch --num_processes=$num_processes --mixed_precision=fp16 \
    "$project_root/mmdiff/train.py" \
    --pretrained_model_name_or_path="Path/to/stable-diffusion-v1-4" \
    --dataset_modality=opt \
    --is_OSI=False \
    --optical_sar_infrared_data_dir="$data_root" \
    --output_dir="$project_root/model-whu/mmdiff-opt" \
    --resolution=512 \
    --train_batch_size=$batch_size \
    --gradient_accumulation_steps=$grad_accum \
    --num_train_epochs=$epochs \
    --max_train_steps=$max_steps \
    --learning_rate=1e-4 \
    --checkpointing_steps=$per_epoch \
    "$@"
