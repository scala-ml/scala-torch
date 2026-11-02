"""Optional independent Diffusers Euler SDXL image-path oracle; fixed noise, no RNG comparison."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, DPMSolverMultistepScheduler
import diffusers

root = Path(sys.argv[1])
torch.set_num_threads(2)
unet = UNet2DConditionModel.from_pretrained(root / "unet", local_files_only=True).eval()
vae = AutoencoderKL.from_pretrained(root / "vae", local_files_only=True).eval()
refs = Path("models/src/test/resources/tiny-sdxl-prompt")
out = Path("models/src/test/resources/tiny-sdxl-pipeline")
out.mkdir(parents=True, exist_ok=True)
manifest = {"cases": [], "tensors": {}, "torch": torch.__version__, "diffusers": diffusers.__version__,
            "model_revision": "fa06b5c3d6d45fabd978486616c6dae068519e85",
            "inputs": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for part in ["unet", "vae"] for p in (root / part).iterdir()
                       if p.suffix in [".json", ".safetensors"]},
            "prompt_manifest_sha256": hashlib.sha256((refs / "manifest.json").read_bytes()).hexdigest()}
def save(name, tensor):
    data = tensor.detach().contiguous().numpy().astype("<f4").tobytes()
    (out / (name + ".bin")).write_bytes(data)
    manifest["tensors"][name] = dict(shape=list(tensor.shape), sha256=hashlib.sha256(data).hexdigest())
def read(name, shape):
    return torch.frombuffer(bytearray((refs / (name + ".bin")).read_bytes()), dtype=torch.float32).reshape(shape)
noise = torch.cos(torch.arange(64, dtype=torch.float32) * .17).reshape(1, 4, 4, 4)
source = torch.sin(torch.arange(192, dtype=torch.float32) * .07).reshape(1, 3, 8, 8)
posterior_noise = torch.sin(torch.arange(64, dtype=torch.float32) * .11).reshape(1, 4, 4, 4)
for name, tensor in [("noise", noise), ("source", source), ("posterior-noise", posterior_noise)]:
    save(name, tensor)
with torch.no_grad():
    for dpm, name, cfg, empty, img, custom in [(dpm, *case) for dpm in [False, True] for case in [("plain", 7.5, False, False, False),
                                         ("empty", 7.5, True, False, False),
                                         ("unguided", 1., False, False, True),
                                         ("img2img", 7.5, False, True, True)]]:
        prefix = "empty" if empty else "plain"
        positive = read(prefix + "-positive", (1, 77, 64))
        negative = read(prefix + "-negative", (1, 77, 64))
        pooled = read(prefix + "-pooled", (1, 32))
        negative_pooled = read(prefix + "-negative-pooled", (1, 32))
        ids = torch.tensor([[12., 16., 1., 2., 8., 8.]] if custom else [[8., 8., 0., 0., 8., 8.]])
        scheduler = DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_order=2) if dpm else EulerDiscreteScheduler()
        scheduler.set_timesteps(2)
        scheduler.timesteps = torch.tensor([500., 250.])
        scheduler.sigmas = torch.tensor([1., .5, 0.])
        start = 1 if img else 0
        scheduler.set_begin_index(start)
        latent = noise.clone()
        if img:
            posterior = vae.encode(source).latent_dist
            clean = (posterior.mean + posterior.std * posterior_noise) * vae.config.scaling_factor
            alpha = 1 / torch.sqrt(1 + scheduler.sigmas[start] ** 2) if dpm else 1
            latent = clean * alpha + noise * (scheduler.sigmas[start] * alpha)
        for i in range(start, 2):
            timestep = scheduler.timesteps[i]
            sample = scheduler.scale_model_input(latent, timestep)
            prediction = unet(sample, timestep, encoder_hidden_states=positive,
                              added_cond_kwargs=dict(text_embeds=pooled, time_ids=ids)).sample
            if cfg > 1:
                neg = unet(sample, timestep, encoder_hidden_states=negative,
                           added_cond_kwargs=dict(text_embeds=negative_pooled, time_ids=ids)).sample
                prediction = neg + cfg * (prediction - neg)
            latent = scheduler.step(prediction, timestep, latent).prev_sample
        name = ("dpm-" if dpm else "") + name
        save(name + "-latent", latent)
        save(name + "-image", vae.decode(latent / vae.config.scaling_factor).sample)
        manifest["cases"].append(dict(name=name, guidance=str(cfg), empty=empty, img2img=img, custom=custom))
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
