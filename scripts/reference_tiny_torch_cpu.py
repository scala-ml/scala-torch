"""Independent Diffusers tiny-SD CPU comparison; not used by the application."""
import json
import sys
from pathlib import Path

import torch
from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
                       EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
                       StableDiffusionPipeline, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageFilter

root, output = map(Path, sys.argv[1:3])
size = int(sys.argv[3]) if len(sys.argv) > 3 else 16
torch.set_num_threads(1)
tokenizer = CLIPTokenizer(vocab_file=str(root / "tokenizer/vocab.json"),
                          merges_file=str(root / "tokenizer/merges.txt"), model_max_length=77)
scheduler = EulerDiscreteScheduler(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
                                   beta_schedule="scaled_linear", timestep_spacing="linspace")
pipe = StableDiffusionPipeline(
    vae=AutoencoderKL.from_pretrained(root / "vae", local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(root / "text_encoder", local_files_only=True),
    tokenizer=tokenizer, unet=UNet2DConditionModel.from_pretrained(root / "unet", local_files_only=True),
    scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to("cpu")
with torch.no_grad():
    result = pipe("a red cat", height=size, width=size, num_inference_steps=3, guidance_scale=1.0,
                  generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
    image = pipe("a red cat", height=size, width=size, num_inference_steps=3, guidance_scale=1.0,
                 generator=torch.Generator(device="cpu").manual_seed(42), output_type="pt").images
    shape = (1, 4, size // pipe.vae_scale_factor, size // pipe.vae_scale_factor)
    low = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(42))
    high = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(1337))
    dot = ((low / torch.norm(low, dim=1, keepdim=True)) *
           (high / torch.norm(high, dim=1, keepdim=True))).sum(1)
    omega, amount = torch.acos(dot), 0.45
    mixed = ((torch.sin((1 - amount) * omega) / torch.sin(omega)).unsqueeze(1) * low +
             (torch.sin(amount * omega) / torch.sin(omega)).unsqueeze(1) * high)
    varied = pipe("a red cat", height=size, width=size, num_inference_steps=3, guidance_scale=7.5,
                  generator=torch.Generator(device="cpu").manual_seed(42), latents=mixed,
                  output_type="latent").images
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    ancestral = pipe("a red cat", height=size, width=size, num_inference_steps=3, guidance_scale=1.0,
                     generator=torch.Generator(device="cpu").manual_seed(42), latents=low.clone(),
                     output_type="latent").images
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                     algorithm_type="dpmsolver++", solver_order=2)
    dpm = pipe("a red cat", height=size, width=size, num_inference_steps=3, guidance_scale=1.0,
               generator=torch.Generator(device="cpu").manual_seed(42), latents=low.clone(),
               output_type="latent").images
    source = ((torch.arange(3 * size * size).reshape(1, 3, size, size) % 256).float() / 127.5) - 1
    imgpipe = StableDiffusionImg2ImgPipeline(**pipe.components).to("cpu")
    img_results = {}
    for name, scheduler_class in (("euler", EulerDiscreteScheduler),
                                  ("ancestral", EulerAncestralDiscreteScheduler),
                                  ("dpm", DPMSolverMultistepScheduler)):
        imgpipe.scheduler = scheduler_class.from_config(scheduler.config)
        img_results[f"img2img_{name}"] = imgpipe("a red cat", image=source, strength=.75,
            num_inference_steps=4, guidance_scale=1.0, generator=torch.Generator(device="cpu").manual_seed(42),
            output_type="latent").images.flatten().tolist()
    imgpipe.scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    img_results["img2img_image"] = imgpipe("a red cat", image=source, strength=.75,
        num_inference_steps=4, guidance_scale=1.0, generator=torch.Generator(device="cpu").manual_seed(42),
        output_type="pt").images.flatten().tolist()
    mask = torch.zeros((1, 1, size, size), dtype=torch.float32)
    mask[:, :, :, size // 2:] = 1
    mask_image = Image.frombytes("L", (size, size), bytes([0 if x < size // 2 else 255
                                                            for y in range(size) for x in range(size)]))
    soft_mask = mask_image.filter(ImageFilter.GaussianBlur(4))
    inpaint_results = {}
    inpaint = StableDiffusionInpaintPipeline(**pipe.components).to("cpu")
    def inpaint_cases(channels):
        for name, scheduler_class in (("euler", EulerDiscreteScheduler),
                                      ("ancestral", EulerAncestralDiscreteScheduler),
                                      ("dpm", DPMSolverMultistepScheduler)):
            inpaint.scheduler = scheduler_class.from_config(scheduler.config)
            for strength in (.75, 1.0):
                result = inpaint("a red cat", image=source, mask_image=mask, height=size, width=size,
                    strength=strength, num_inference_steps=4, guidance_scale=1.0,
                    generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
                inpaint_results[f"inpaint_{channels}_{name}_{int(strength * 100)}"] = result.flatten().tolist()
                if name == "euler" and strength == .75:
                    generated = inpaint("a red cat", image=source, mask_image=mask, height=size, width=size,
                        strength=strength, num_inference_steps=4, guidance_scale=1.0,
                        generator=torch.Generator(device="cpu").manual_seed(42), output_type="pt").images
                    overlaid = generated * mask + ((source + 1) / 2) * (1 - mask)
                    inpaint_results[f"inpaint_{channels}_image"] = overlaid.flatten().tolist()
        inpaint.scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
        soft = inpaint("a red cat", image=source, mask_image=soft_mask, height=size, width=size,
            strength=.75, num_inference_steps=4, guidance_scale=1.0,
            generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
        inpaint_results[f"inpaint_{channels}_soft"] = soft.flatten().tolist()
        soft_image = inpaint("a red cat", image=source, mask_image=soft_mask, height=size, width=size,
            strength=.75, num_inference_steps=4, guidance_scale=1.0,
            generator=torch.Generator(device="cpu").manual_seed(42), output_type="pt").images
        alpha = torch.tensor(list(soft_mask.getdata()), dtype=torch.float32).reshape(1, 1, size, size) / 255
        inpaint_results[f"inpaint_{channels}_soft_image"] = (soft_image * alpha + ((source + 1) / 2) * (1 - alpha)).flatten().tolist()
    inpaint_cases(4)
    old = inpaint.unet.conv_in
    widened = torch.nn.Conv2d(9, old.out_channels, old.kernel_size, old.stride, old.padding)
    widened.weight[:, :4] = old.weight
    for channel in range(4, 9):
        widened.weight[:, channel].fill_(.025 * (channel - 3))
    widened.bias.copy_(old.bias)
    inpaint.unet.conv_in = widened
    inpaint.unet.register_to_config(in_channels=9)
    inpaint_cases(9)
output.write_text(json.dumps(dict(seed=42, shape=list(result.shape), latent=result.flatten().tolist(),
                                  image=image.flatten().tolist(), variation_latent=varied.flatten().tolist(),
                                  ancestral_latent=ancestral.flatten().tolist(), dpm_latent=dpm.flatten().tolist(),
                                  soft_mask=list(soft_mask.getdata()), **img_results, **inpaint_results)),
                  encoding="utf-8")
