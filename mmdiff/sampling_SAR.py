"""SAR image generation guided by OPT attention/resnet features.

This script loads the SAR LoRA adapter on top of the shared stable-diffusion
backbone and injects the features extracted during the OPT run.
"""

import os
import random
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import SamplingConfig
from register import (
    AttentionStore,
    ResidualStore,
    register_attention_control,
    register_resnet_control,
    register_single_channel_decoder,
)
from utils import load_model

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    cfg = SamplingConfig()
    set_seed(cfg.seed)
    output_path = os.path.join(cfg.output_dir, "sar")
    weight_dtype = torch.bfloat16

    # ---------------- Load diffusion model ----------------
    models = load_model(model_path=cfg.model_path)
    vae = models["vae"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    unet = models["unet"]
    scheduler = models["scheduler"]

    # Load SAR LoRA weights.
    unet.load_attn_procs(cfg.sar_lora_path)

    torch_device = torch.device(cfg.device)
    vae.to(torch_device, dtype=weight_dtype)
    text_encoder.to(torch_device, dtype=weight_dtype)
    unet.to(torch_device, dtype=weight_dtype)

    # ---------------- Text embeddings ----------------
    generator = torch.Generator(device=torch_device).manual_seed(cfg.seed)

    text_input = tokenizer(
        [cfg.prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * cfg.batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(torch_device)
        )[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # ---------------- Initialize latents ----------------
    latents = torch.randn(
        (
            cfg.batch_size,
            unet.config.in_channels,
            cfg.resolution // 8,
            cfg.resolution // 8,
        ),
        generator=generator,
        device=torch_device,
        dtype=weight_dtype,
    )
    scheduler.set_timesteps(cfg.num_step)
    timesteps = scheduler.timesteps.tolist()

    # ---------------- ResNet feature injection ----------------
    residual_controller = ResidualStore(cfg.resnet_path)
    register_resnet_control(unet, residual_controller)
    residual_controller.set_save_mode(False)
    residual_controller.set_replace_mode(False)

    t_half = timesteps[: int(len(timesteps) * cfg.resnet_time)]
    residual_controller.inject_residuals_for_range(
        timesteps=t_half,
        place_in_unet="up",
        layer_range=[cfg.resnet_layer],
        device=torch_device,
    )

    # ---------------- Attention feature injection ----------------
    attn_controller = AttentionStore(cfg.attn_path)
    register_attention_control(unet, attn_controller)
    attn_controller.set_timesteps_to_inject(timesteps)
    attn_controller.set_save_mode(False)
    attn_controller.set_replace_mode(True)

    attn_layer = [int(item) for item in cfg.attn_layer.split(",")]
    attn_controller.set_replace_layers(
        attn_layers=attn_layer, components=[cfg.component]
    )

    latents = latents.clone()
    register_single_channel_decoder(vae)

    # ---------------- Diffusion inference ----------------
    for _i, t in enumerate(tqdm(scheduler.timesteps)):
        residual_controller.set_timestep(int(t))
        attn_controller.set_timestep(int(t))

        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, timestep=t)

        with torch.no_grad():
            noise_pred = unet(
                latent_input, t, encoder_hidden_states=text_embeddings
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # ---------------- Decode and save image ----------------
    latents_scaled = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents_scaled).sample

    image = (
        (image / 2 + 0.5)
        .clamp(0, 1)
        .mul(255)
        .byte()[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    if image.shape[-1] == 1:
        image = image.squeeze(-1)

    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f"{cfg.n}.png")
    Image.fromarray(image).save(output_path)

    print(f"Generated image saved to {output_path}")


if __name__ == "__main__":
    main()
