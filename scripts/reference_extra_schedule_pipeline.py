"""Independent NUI/Diffusers tiny-SD schedule output fixture; not a runtime dependency."""
import json
import sys
from pathlib import Path
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

model, nui, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(nui / "python"))
from sdfx.schedulers import build_scheduler, plan

torch.set_num_threads(1)
base = dict(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
            beta_schedule="scaled_linear", timestep_spacing="linspace")
initial, _ = build_scheduler(plan("Euler", "Normal"), base)
pipe = StableDiffusionPipeline(
    vae=AutoencoderKL.from_pretrained(model / "vae", local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(model / "text_encoder", local_files_only=True),
    tokenizer=CLIPTokenizer(vocab_file=str(model / "tokenizer/vocab.json"),
                            merges_file=str(model / "tokenizer/merges.txt"), model_max_length=77),
    unet=UNet2DConditionModel.from_pretrained(model / "unet", local_files_only=True),
    scheduler=initial, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to("cpu")
imgpipe = StableDiffusionImg2ImgPipeline(**pipe.components).to("cpu")
source = ((torch.arange(3 * 16 * 16).reshape(1, 3, 16, 16) % 256).float() / 127.5) - 1
results = []
with torch.no_grad():
    for sampler in ("Euler", "Euler a", "DPM++ 2M"):
        for schedule in ("Normal", "Simple", "SGM Uniform", "Polyexponential"):
            pipe.scheduler, _ = build_scheduler(plan(sampler, schedule), base)
            # NUI creates the initial noise separately from its freshly seeded scheduler generator.
            side = 16 // pipe.vae_scale_factor
            initial_noise = torch.randn((1, 4, side, side), generator=torch.Generator(device="cpu").manual_seed(42))
            latent = pipe("a red cat", height=16, width=16, num_inference_steps=4, guidance_scale=1,
                          latents=initial_noise, generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
            imgpipe.scheduler, _ = build_scheduler(plan(sampler, schedule), base)
            img = imgpipe("a red cat", image=source, strength=.75, num_inference_steps=4, guidance_scale=1,
                          generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
            pipe.scheduler, _ = build_scheduler(plan(sampler, schedule), base)
            initial_noise = torch.randn((1, 4, side, side), generator=torch.Generator(device="cpu").manual_seed(42))
            guided = pipe("a red cat", negative_prompt="blurry", height=16, width=16,
                          num_inference_steps=4, guidance_scale=7.5, latents=initial_noise,
                          generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
            imgpipe.scheduler, _ = build_scheduler(plan(sampler, schedule), base)
            guided_img = imgpipe("a red cat", negative_prompt="blurry", image=source, strength=.75,
                                 num_inference_steps=4, guidance_scale=7.5,
                                 generator=torch.Generator(device="cpu").manual_seed(42), output_type="latent").images
            results.append(dict(sampler=sampler, schedule=schedule, txt=latent.flatten().tolist(), img=img.flatten().tolist(),
                                guided_txt=guided.flatten().tolist(), guided_img=guided_img.flatten().tolist()))
output.write_text(json.dumps(results), encoding="utf-8")
with torch.no_grad():
    # Match NUI Latent Upscale: sample source-sized posterior first, then resize;
    # continue the same generator for img2img noise and ancestral steps.
    cases = {}
    for sampler in ("Euler", "Euler a", "DPM++ 2M"):
        imgpipe.scheduler, _ = build_scheduler(plan(sampler, "Normal"), base)
        generator = torch.Generator(device="cpu").manual_seed(42)
        posterior = pipe.vae.encode(source).latent_dist.sample(generator=generator) * pipe.vae.config.scaling_factor
        enlarged = torch.nn.functional.interpolate(posterior, size=(12, 12), mode="nearest-exact")
        result = imgpipe("a red cat", image=enlarged, strength=.75, num_inference_steps=4, guidance_scale=1,
                         generator=generator, output_type="latent").images
        cases[sampler] = result.flatten().tolist()
    output.with_name("source-latent-upscale-reference.json").write_text(json.dumps(cases), encoding="utf-8")
    inpaint = StableDiffusionInpaintPipeline(**pipe.components).to("cpu")
    source_mask = torch.zeros((1, 1, 16, 16)); source_mask[..., :, 8:] = 1
    target_mask = torch.zeros((1, 1, 24, 24)); target_mask[..., :, 12:] = 1
    cases = {}
    for sampler in ("Euler", "Euler a", "DPM++ 2M"):
        inpaint.scheduler, _ = build_scheduler(plan(sampler, "Normal"), base)
        generator = torch.Generator(device="cpu").manual_seed(42)
        def encode(pixels):
            latent = pipe.vae.encode(pixels).latent_dist.sample(generator=generator) * pipe.vae.config.scaling_factor
            return torch.nn.functional.interpolate(latent, size=(12, 12), mode="nearest-exact")
        clean, masked = encode(source), encode(source * (source_mask < .5))
        result = inpaint("a red cat", image=clean, mask_image=target_mask, masked_image_latents=masked,
                         height=24, width=24, strength=.75, num_inference_steps=4, guidance_scale=1,
                         generator=generator, output_type="latent").images
        cases[sampler] = result.flatten().tolist()
    output.with_name("masked-latent-upscale-reference.json").write_text(json.dumps(cases), encoding="utf-8")
    cases = {}
    for sampler in ("Euler", "Euler a", "DPM++ 2M"):
        inpaint.scheduler, _ = build_scheduler(plan(sampler, "Normal"), base)
        generator = torch.Generator(device="cpu").manual_seed(42)
        clean, masked = encode(source), encode(source * (source_mask < .5))
        result = inpaint("a red cat", image=clean, mask_image=target_mask, masked_image_latents=masked,
                         height=24, width=24, strength=1.0, num_inference_steps=4, guidance_scale=1,
                         generator=generator, output_type="latent").images
        cases[sampler] = result.flatten().tolist()
    output.with_name("masked-latent-upscale-full-reference.json").write_text(json.dumps(cases), encoding="utf-8")
    old = inpaint.unet.conv_in
    widened = torch.nn.Conv2d(9, old.out_channels, old.kernel_size, old.stride, old.padding)
    widened.weight[:, :4] = old.weight
    for channel in range(4, 9):
        widened.weight[:, channel].fill_(.025 * (channel - 3))
    widened.bias.copy_(old.bias)
    inpaint.unet.conv_in = widened
    inpaint.unet.register_to_config(in_channels=9)
    cases = {}
    for sampler in ("Euler", "Euler a", "DPM++ 2M"):
        inpaint.scheduler, _ = build_scheduler(plan(sampler, "Normal"), base)
        generator = torch.Generator(device="cpu").manual_seed(42)
        def encode9(pixels):
            latent = pipe.vae.encode(pixels).latent_dist.sample(generator=generator) * pipe.vae.config.scaling_factor
            return torch.nn.functional.interpolate(latent, size=(12, 12), mode="nearest-exact")
        clean, masked = encode9(source), encode9(source * (source_mask < .5))
        result = inpaint("a red cat", image=clean, mask_image=target_mask, masked_image_latents=masked,
                         height=24, width=24, strength=.75, num_inference_steps=4, guidance_scale=1,
                         generator=generator, output_type="latent").images
        cases[sampler] = result.flatten().tolist()
    output.with_name("masked-latent-upscale-9-reference.json").write_text(json.dumps(cases), encoding="utf-8")
    cases = {}
    for sampler in ("Euler", "Euler a", "DPM++ 2M"):
        inpaint.scheduler, _ = build_scheduler(plan(sampler, "Normal"), base)
        generator = torch.Generator(device="cpu").manual_seed(42)
        clean, masked = encode9(source), encode9(source * (source_mask < .5))
        result = inpaint("a red cat", image=clean, mask_image=target_mask, masked_image_latents=masked,
                         height=24, width=24, strength=1.0, num_inference_steps=4, guidance_scale=1,
                         generator=generator, output_type="latent").images
        cases[sampler] = result.flatten().tolist()
    output.with_name("masked-latent-upscale-9-full-reference.json").write_text(json.dumps(cases), encoding="utf-8")
    cases = {}
    for strength in (.75, 1.0):
        for sampler in ("Euler", "Euler a", "DPM++ 2M"):
            inpaint.scheduler, _ = build_scheduler(plan(sampler, "Normal"), base)
            generator = torch.Generator(device="cpu").manual_seed(42)
            clean, masked = encode9(source), encode9(torch.zeros_like(source))
            result = inpaint("a red cat", image=clean, mask_image=torch.ones_like(target_mask), masked_image_latents=masked,
                             height=24, width=24, strength=strength, num_inference_steps=4, guidance_scale=1,
                             generator=generator, output_type="latent").images
            cases[f"{sampler}:{strength}"] = result.flatten().tolist()
    output.with_name("unmasked-inpaint-latent-upscale-reference.json").write_text(json.dumps(cases), encoding="utf-8")
