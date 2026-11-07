"""NUI fallback scheduler reference for Euler a and DPM++ 2M."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
from sdfx.schedulers import build_scheduler, plan

base = dict(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
            beta_schedule="scaled_linear", timestep_spacing="leading", steps_offset=1)
cases = []
for sampler in ("Euler a", "DPM++ 2M"):
    for schedule in ("Normal", "Simple", "SGM Uniform"):
        for count in (1, 3, 4, 9):
            scheduler, warnings = build_scheduler(plan(sampler, schedule), base)
            scheduler.set_timesteps(count)
            cases.append(dict(sampler=sampler, schedule=schedule, count=count,
                              timesteps=scheduler.timesteps.tolist(), sigmas=scheduler.sigmas.tolist(), warnings=warnings))
print(json.dumps(cases))
