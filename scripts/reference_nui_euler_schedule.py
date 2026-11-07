"""Generate NUI/Diffusers Euler schedule fixtures; not used by the runtime.

Usage: python scripts/reference_nui_euler_schedule.py NUI_ROOT
"""
import json
import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
import diffusers
from sdfx.schedulers import build_scheduler, plan

base = dict(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
            beta_schedule="scaled_linear", timestep_spacing="leading", steps_offset=1)
cases = []
for schedule in ("Uniform", "Karras", "Exponential", "Normal", "Simple", "SGM Uniform"):
    for count in (1, 3, 4, 9):
        scheduler, warnings = build_scheduler(plan("Euler", schedule), base)
        scheduler.set_timesteps(count)
        cases.append(dict(schedule=schedule, count=count, timesteps=scheduler.timesteps.tolist(),
            sigmas=scheduler.sigmas.tolist(), initial_noise_sigma=float(scheduler.init_noise_sigma), warnings=warnings))
path = Path("models/src/test/resources/nui-euler-schedule-reference.json")
sources = {name: hashlib.sha256((root / "python/sdfx" / name).read_bytes()).hexdigest()
           for name in ("schedulers.py", "sigmas.py")}
path.write_text(json.dumps(dict(diffusers=diffusers.__version__, sources=sources, cases=cases), indent=2) + "\n", encoding="utf-8", newline="\n")
print(path)
