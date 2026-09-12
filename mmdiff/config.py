"""Centralized configuration for MMDiff sampling and training pipelines."""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SamplingConfig:
    """All inference parameters for the OPT, SAR and IR pipelines."""

    # Path to the pre-trained OPT diffusion model.
    model_path: str = "./models/MMDiff/mmdiff-opt"

    # Paths to the final SAR and IR LoRA adapters.
    sar_lora_path: str = "./models/MMDiff/mmdiff-sar/sd-sar-lora-ship"
    ir_lora_path: str = "./models/MMDiff/mmdiff-ir/sd-ir-lora-ship"

    # Directories where the OPT run stores / reads attention and resnet features.
    attn_path: str = "./features/visible_attn_maps"
    resnet_path: str = "./features/visible_resnet_maps"

    # Reproducibility.
    seed: int = 2026
    output_dir: str = "result"

    # Diffusion sampling.
    batch_size: int = 1
    device: str = "cuda:0"
    n: str = "ship" # Identifier used as the generated image filename.
    resolution: int = 256
    num_step: int = 50
    prompt: str = "There is a ship in the blue water on the shore ."
    guidance_scale: float = 7.5

    # feature setting.
    resnet_layer: int = 2 # Using the third resnet layer for injection.
    resnet_time: float = 1.0 # Ratio of timesteps during which resnet features are injected.
    attn_layer: str = "1,2,3,4,5,6,7,8,9" # Using all query feature for injection.
    component: str = "q" # Self-attention component to inject: q.

@dataclass
class BaseTrainingConfig:
    # ============================================================
    # Model and Dataset
    # ============================================================

    # Pretrained model to use for training
    pretrained_model_name_or_path: str = "path/to/stable-diffusion-v1-4"

    # Model revision and variant
    revision: Optional[str] = None
    variant: Optional[str] = None
    # Kept separate for compatibility with non-EMA checkpoints used by UNet.
    non_ema_revision: Optional[str] = None

    # Dataset
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    cache_dir: Optional[str] = None

    # Dataset columns
    image_column: str = "image"
    caption_column: str = "text"

    # Limit the number of training samples
    max_train_samples: Optional[int] = None

    # ============================================================
    # Input / Prompt
    # ============================================================

    # Input perturbation strength
    input_perturbation: float = 0.0

    # Validation prompts
    validation_prompts: Optional[List[str]] = None

    # ============================================================
    # Output and Logging
    # ============================================================

    output_dir: str = "mmdiff-opt"
    logging_dir: str = "logs"

    # Random seed
    seed: Optional[int] = None

    # ============================================================
    # Image Processing
    # ============================================================

    resolution: int = 256
    center_crop: bool = True
    random_flip: bool = True

    # ============================================================
    # Training
    # ============================================================

    train_batch_size: int = 1
    num_train_epochs: int = 10
    max_train_steps: Optional[int] = 15000

    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True

    # ============================================================
    # Optimizer
    # ============================================================

    learning_rate: float = 1e-4
    scale_lr: bool = False

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8

    max_grad_norm: float = 1.0

    # ============================================================
    # Learning Rate Scheduler
    # ============================================================

    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0

    # ============================================================
    # Noise / Diffusion Training
    # ============================================================

    snr_gamma: Optional[float] = None
    noise_offset: float = 0.0

    prediction_type: Optional[str] = None

    # Dream Training
    dream_training: bool = False
    dream_detail_preservation: float = 1.0

    # ============================================================
    # Memory and Performance
    # ============================================================

    use_8bit_adam: bool = False
    allow_tf32: bool = False

    # Enable xFormers memory-efficient attention
    enable_xformers_memory_efficient_attention: bool = False

    # ============================================================
    # EMA
    # ============================================================

    use_ema: bool = False
    offload_ema: bool = False
    foreach_ema: bool = False

    # ============================================================
    # DataLoader
    # ============================================================

    dataloader_num_workers: int = 0

    # ============================================================
    # Checkpointing
    # ============================================================

    checkpointing_steps: int = 5000
    checkpoints_total_limit: Optional[int] = None

    # Resume training from a checkpoint
    resume_from_checkpoint: Optional[str] = None

    # ============================================================
    # Mixed Precision / Logging
    # ============================================================

    mixed_precision: Optional[str] = "fp16"
    report_to: str = "wandb"

    # ============================================================
    # Distributed Training
    # ============================================================

    local_rank: int = -1

    # ============================================================
    # Validation
    # ============================================================

    validation_epochs: int = 1

    # ============================================================
    # Experiment Tracking
    # ============================================================

    tracker_project_name: str = "mmdiff"


@dataclass
class OSIConfig(BaseTrainingConfig):
    """Training configuration specific to the Optical-SAR-Infrared dataset."""

    # ``dataset_modality`` selects the corresponding {opt,sar,ir}_train.jsonl annotation file.
    optical_sar_infrared_data_dir: str = "/media/ubuntun/hdd/dataset/Optical-SAR-Infrared"
    dataset_modality: str = "opt"
    # SAR and IR LoRA runs train one semantic category at a time (for example,ship). 
    # OPT uses all categories and leaves this unset.
    dataset_category: Optional[str] = None
