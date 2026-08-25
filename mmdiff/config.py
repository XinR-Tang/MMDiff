"""Centralized configuration for the OPT, SAR and IR sampling pipelines."""

from dataclasses import dataclass

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
