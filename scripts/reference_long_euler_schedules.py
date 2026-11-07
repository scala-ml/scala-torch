"""NUI's Euler schedules above the training step count (test fixture only)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
from sdfx.schedulers import build_scheduler, plan

base = dict(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
            beta_schedule="scaled_linear", timestep_spacing="linspace")
cases = []
for name in ("Uniform", "Normal", "Simple", "SGM Uniform", "Karras", "Exponential", "Polyexponential"):
    for count in (1001, 2000, 10000):
        scheduler, _ = build_scheduler(plan("Euler", name), base)
        scheduler.set_timesteps(count)
        indices = sorted(set((0, 1, count // 3, count // 2, count - 2, count - 1)))
        cases.append(dict(schedule=name, count=count, indices=indices,
                          timesteps=[float(scheduler.timesteps[i]) for i in indices],
                          sigmas=[float(scheduler.sigmas[i]) for i in indices]))
Path("models/src/test/resources/long-euler-schedules-reference.json").write_text(json.dumps(cases), encoding="utf-8")
