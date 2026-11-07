"""Regenerate small Diffusers Euler-ancestral step fixtures (not a runtime dependency)."""
import json
from pathlib import Path
import torch
import diffusers
from diffusers import EulerAncestralDiscreteScheduler

cases = []
for prediction in ("epsilon", "v_prediction"):
    scheduler = EulerAncestralDiscreteScheduler(num_train_timesteps=10, beta_start=.001,
        beta_end=.02, beta_schedule="scaled_linear", prediction_type=prediction)
    scheduler.set_timesteps(4)
    generator = torch.Generator().manual_seed(171)
    noises = torch.Generator().manual_seed(171)
    sample = torch.tensor([.3, -.7, 1.2, -.1], dtype=torch.float32)
    trajectory = []
    for i, t in enumerate(scheduler.timesteps):
        output = torch.tensor([.1 + i*.07, -.2 + i*.03, .4 - i*.06, .5 + i*.02], dtype=torch.float32)
        noise = torch.randn(sample.shape, generator=noises)
        result = scheduler.step(output, t, sample, generator=generator)
        trajectory.append(dict(sample=sample.tolist(), model_output=output.tolist(), noise=noise.tolist(),
            previous=result.prev_sample.tolist(), original=result.pred_original_sample.tolist()))
        sample = result.prev_sample
    cases.append(dict(prediction=prediction, timesteps=scheduler.timesteps.tolist(),
        sigmas=scheduler.sigmas.tolist(), initial_noise_sigma=float(scheduler.init_noise_sigma), steps=trajectory))
path = Path("models/src/test/resources/euler-ancestral-reference.json")
path.write_text(json.dumps(dict(diffusers=diffusers.__version__, cases=cases), indent=2) + "\n", encoding="utf-8", newline="\n")
print(path)
