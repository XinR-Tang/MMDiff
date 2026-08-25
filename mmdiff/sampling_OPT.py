"""OPT (visible-light) image generation.

This script runs the visible-light diffusion model and simultaneously extracts
the attention and resnet features that are later consumed by the SAR and IR
pipelines.
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
    output_path = os.path.join(cfg.output_dir, "opt")
    weight_dtype = torch.bfloat16

    # ---------------- Load visible-light diffusion model ----------------
    models = load_model(model_path=cfg.model_path)
    vae = models["vae"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    unet = models["unet"]
    scheduler = models["scheduler"]

    torch_device = torch.device(cfg.device)
    vae.to(torch_device, dtype=weight_dtype)
    text_encoder.to(torch_device, dtype=weight_dtype)
    unet.to(torch_device, dtype=weight_dtype)

    # ---------------- Initialize latents ----------------
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

    # ---------------- Register both Attention and ResNet controllers ----------------
    # AttentionStore hooks the Attention modules; ResidualStore hooks the
    # ResnetBlock2D modules. They act on different submodules with non-overlapping
    # forward wrappers, so both features can be extracted simultaneously in a
    # single UNet forward pass without conflict.
    attn_controller = AttentionStore(cfg.attn_path)
    register_attention_control(unet, attn_controller)
    attn_controller.set_save_mode(True)
    attn_controller.set_replace_mode(False)

    resnet_controller = ResidualStore(cfg.resnet_path)
    register_resnet_control(unet, resnet_controller)
    resnet_controller.set_save_mode(True)
    resnet_controller.set_replace_mode(False)

    # ---------------- Diffusion model inference ----------------
    for _i, t in enumerate(tqdm(scheduler.timesteps)):
        attn_controller.set_timestep(t)
        resnet_controller.set_timestep(t)

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

    # ---------------- Decode and save the image ----------------
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

    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f"{cfg.n}.png")
    Image.fromarray(image).save(output_path)

    print(f"Generated image saved to {output_path}")
    print(f"Attention features saved to {cfg.attn_path}")
    print(f"ResNet features saved to {cfg.resnet_path}")


if __name__ == "__main__":
    main()
