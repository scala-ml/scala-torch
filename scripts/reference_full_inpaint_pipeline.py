"""Build independent NUI/PyTorch full-resolution inpainting image reference."""
import json
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

model, nui, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(nui / "python"))
from sdfx.inpaint import prepare

torch.set_num_threads(1)
source = Image.new("RGB", (80, 64))
source.putdata([((x * 3 + y * 5) % 256,) * 3 for y in range(64) for x in range(80)])
mask = Image.new("L", source.size)
mask.putdata([255 if 28 <= x < 52 and 20 <= y < 44 else 0 for y in range(64) for x in range(80)])
job = prepare(source, mask, 64, 64, mask_blur=2, inpainting_fill=1, full_res=True, full_res_padding=4)
scheduler = EulerDiscreteScheduler(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
                                    beta_schedule="scaled_linear", timestep_spacing="linspace")
pipe = StableDiffusionInpaintPipeline(
    vae=AutoencoderKL.from_pretrained(model / "vae", local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(model / "text_encoder", local_files_only=True),
    tokenizer=CLIPTokenizer(vocab_file=str(model / "tokenizer/vocab.json"),
                            merges_file=str(model / "tokenizer/merges.txt"), model_max_length=77),
    unet=UNet2DConditionModel.from_pretrained(model / "unet", local_files_only=True),
    scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to("cpu")
with torch.no_grad():
    generated = pipe("a red cat", image=job.image, mask_image=job.mask, height=64, width=64,
                     strength=.75, num_inference_steps=4, guidance_scale=1.0,
                     generator=torch.Generator(device="cpu").manual_seed(42), output_type="pil").images[0]
result = job.finalize(generated)
output.write_text(json.dumps(dict(width=80, height=64, box=job.paste_box,
                                  rgb=[v for p in result.getdata() for v in p])), encoding="utf-8")
