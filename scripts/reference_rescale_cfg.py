"""Generate independent Diffusers tiny-SD CFG-rescale latent references."""
import json
import sys
from pathlib import Path
import torch
from diffusers import (AutoencoderKL, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

root, output = map(Path, sys.argv[1:3])
torch.set_num_threads(1)
tokenizer = CLIPTokenizer(vocab_file=str(root / 'tokenizer/vocab.json'),
                          merges_file=str(root / 'tokenizer/merges.txt'), model_max_length=77)
scheduler = EulerDiscreteScheduler(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
                                   beta_schedule='scaled_linear', timestep_spacing='linspace')
pipe = StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root / 'vae', local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(root / 'text_encoder', local_files_only=True), tokenizer=tokenizer,
    unet=UNet2DConditionModel.from_pretrained(root / 'unet', local_files_only=True), scheduler=scheduler,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False).to('cpu')
imgpipe = StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
source = ((torch.arange(3 * 16 * 16).reshape(1, 3, 16, 16) % 256).float() / 127.5) - 1
cases = {}
with torch.no_grad():
    for name, cls in [('euler', EulerDiscreteScheduler), ('ancestral', EulerAncestralDiscreteScheduler),
                      ('dpm', DPMSolverMultistepScheduler)]:
        for mode, runner in [('txt2img', pipe), ('img2img', imgpipe)]:
            runner.scheduler = cls.from_config(scheduler.config)
            args = dict(prompt='a red cat', num_inference_steps=4, guidance_scale=7.5, guidance_rescale=.7,
                        generator=torch.Generator(device='cpu').manual_seed(42), output_type='latent')
            if mode == 'img2img': args.update(image=source, strength=.75)
            else:
                # NUI draws initial txt2img noise in its own seeded generator;
                # scheduler noise starts from a separate generator at the same seed.
                args.update(height=16, width=16, latents=torch.randn((1, 4, 8, 8),
                    generator=torch.Generator(device='cpu').manual_seed(42)))
            cases[f'{mode}_{name}'] = runner(**args).images.flatten().tolist()
output.write_text(json.dumps(cases), encoding='utf-8')
