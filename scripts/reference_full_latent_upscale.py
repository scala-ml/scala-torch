"""Independent NUI/PyTorch full-resolution masked Latent Upscale fixture."""
import json
import sys
from pathlib import Path
import torch
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

model, nui, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(nui / "python"))
from sdfx.inpaint import prepare
from sdfx.latent_upscale import full_steps
from sdfx.schedulers import build_scheduler, plan

torch.set_num_threads(1)
source = Image.new("RGB", (20, 16))
source.putdata([((x * 11 + y * 7) % 256,) * 3 for y in range(16) for x in range(20)])
mask = Image.new("L", source.size)
mask.putdata([255 if 7 <= x < 13 and 5 <= y < 11 else 0 for y in range(16) for x in range(20)])
job = prepare(source, mask, 16, 16, mask_blur=1, inpainting_fill=1, full_res=True, full_res_padding=2)
base = dict(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
            beta_schedule="scaled_linear", timestep_spacing="linspace")
scheduler, _ = build_scheduler(plan("Euler", "Normal"), base)
pipe = StableDiffusionInpaintPipeline(
    vae=AutoencoderKL.from_pretrained(model / "vae", local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(model / "text_encoder", local_files_only=True),
    tokenizer=CLIPTokenizer(vocab_file=str(model / "tokenizer/vocab.json"),
                            merges_file=str(model / "tokenizer/merges.txt"), model_max_length=77),
    unet=UNet2DConditionModel.from_pretrained(model / "unet", local_files_only=True),
    scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to("cpu")
with torch.no_grad():
    generator = torch.Generator(device="cpu").manual_seed(42)
    pixels = pipe.image_processor.preprocess(job.image)
    mask_pixels = torch.tensor(list(job.mask.getdata()), dtype=torch.float32).reshape(1, 1, 16, 16) / 255
    def encode(pixels):
        latent = pipe.vae.encode(pixels).latent_dist.sample(generator=generator) * pipe.vae.config.scaling_factor
        return torch.nn.functional.interpolate(latent, size=(16 // pipe.vae_scale_factor,) * 2, mode="nearest-exact")
    clean = encode(pixels)
    masked = encode(pixels * (mask_pixels < .5))
    count, strength, _ = full_steps(2, .5)
    generated = pipe("a red cat", image=clean, mask_image=job.mask, masked_image_latents=masked,
                     height=16, width=16, strength=strength, num_inference_steps=count, guidance_scale=1,
                     generator=generator, output_type="pil").images[0]
result = job.finalize(generated)
output.write_text(json.dumps(dict(width=20, height=16, box=job.paste_box,
                                  rgb=[v for pixel in result.getdata() for v in pixel])), encoding="utf-8")
