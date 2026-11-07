"""Capture NUI/Diffusers schedule behaviour; never used by the native runtime."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
import diffusers
from sdfx.schedulers import build_scheduler, plan

profiles = [
    dict(num_train_timesteps=1000, beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear"),
    dict(num_train_timesteps=1000, beta_start=.0001, beta_end=.02, beta_schedule="linear"),
    dict(num_train_timesteps=8, trained_betas=[.001, .002, .003, .004, .005, .006, .007, .008]),
]
cases = []
for profile, config in enumerate(profiles):
    base = dict(config, timestep_spacing="leading", steps_offset=1)
    for sampler in ("DPM++ 2M", "Euler a"):
      for count in ((1, 3, 4, 7, 9) if profile != 2 else (1, 3, 4, 7)):
        scheduler, warnings = build_scheduler(plan(sampler, "Exponential"), base)
        scheduler.set_timesteps(count)
        cases.append(dict(profile=profile, sampler=sampler, count=count, timesteps=scheduler.timesteps.tolist(),
                          sigmas=scheduler.sigmas.tolist(), warnings=warnings))
path = Path("models/src/test/resources/nui-dpm-exponential-reference.json")
source = hashlib.sha256((root / "python/sdfx/schedulers.py").read_bytes()).hexdigest()
path.write_text(json.dumps(dict(diffusers=diffusers.__version__, source=source, cases=cases), indent=2) + "\n", encoding="utf-8", newline="\n")
print(path)
