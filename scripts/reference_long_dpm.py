"""Inspect NUI DPM++ 2M long-schedule behavior; test fixture only."""
import json
import sys
from pathlib import Path
import torch
from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

model, nui, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(nui / "python"))
from sdfx.schedulers import build_scheduler, plan

torch.set_num_threads(1)
base = dict(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
            beta_schedule="scaled_linear", timestep_spacing="linspace")
scheduler, _ = build_scheduler(plan("DPM++ 2M", "Normal"), base)
pipe = StableDiffusionImg2ImgPipeline(
    vae=AutoencoderKL.from_pretrained(model / "vae", local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(model / "text_encoder", local_files_only=True),
    tokenizer=CLIPTokenizer(vocab_file=str(model / "tokenizer/vocab.json"),
                            merges_file=str(model / "tokenizer/merges.txt"), model_max_length=77),
    unet=UNet2DConditionModel.from_pretrained(model / "unet", local_files_only=True),
    scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to("cpu")
source = ((torch.arange(3 * 16 * 16).reshape(1, 3, 16, 16) % 256).float() / 127.5) - 1
cases = []
with torch.no_grad():
    for schedule in ("Normal", "Simple", "Karras", "Exponential", "Polyexponential"):
        for count, strength in ((2000,.0015),(2000,.002),(10000,.0002),(10000,.0003)):
            pipe.scheduler, warnings = build_scheduler(plan("DPM++ 2M", schedule), base)
            result = pipe("a red cat", image=source, strength=strength, num_inference_steps=count, guidance_scale=1,
                          generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
            finite = bool(torch.isfinite(result).all())
            cases.append(dict(schedule=schedule, count=count, strength=strength, finite=finite,
                              latent=result.flatten().tolist() if finite else None, warnings=warnings))
output.write_text(json.dumps(cases), encoding="utf-8")
print([(r["schedule"],r["count"],r["strength"],r["finite"]) for r in cases])
