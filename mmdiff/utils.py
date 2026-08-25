import torch
from pathlib import Path
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def load_model(model_path: str | Path) -> dict:
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", use_safetensors=True)
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    return {
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "scheduler": scheduler
    }
