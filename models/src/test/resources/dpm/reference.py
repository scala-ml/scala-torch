"""Offline fixture generator only; Scala tests and inference do not run Python.

Reference: diffusers 0.33.1 DPMSolverMultistepScheduler, torch 2.7.1 CPU.
Run with one output TSV path. Uses NUI's default DPM++ 2M/Karras profile.
"""
import sys
from pathlib import Path
import torch
import diffusers
from diffusers import DPMSolverMultistepScheduler

assert diffusers.__version__ == "0.33.1"
assert torch.__version__.split("+")[0] == "2.7.1"
rows = ["prediction\tstep\ttimestep\tsigma\tnext_sigma\tx0\tx1\tm0\tm1\ty0\ty1"]
for prediction in ("epsilon", "sample", "v_prediction"):
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", algorithm_type="dpmsolver++",
        solver_order=2, solver_type="midpoint", use_karras_sigmas=not (len(sys.argv) > 2 and sys.argv[2] == "uniform"),
        prediction_type=prediction, final_sigmas_type="zero",
    )
    scheduler.set_timesteps(4)
    sample = torch.tensor([0.25, -0.75], dtype=torch.float32)
    for i, timestep in enumerate(scheduler.timesteps):
        model = torch.tensor([0.1 * (i + 1), -0.2 * (i + 1)], dtype=torch.float32)
        result = scheduler.step(model, timestep, sample).prev_sample
        values = [prediction, i, int(timestep), float(scheduler.sigmas[i]),
                  float(scheduler.sigmas[i + 1]), *sample.tolist(), *model.tolist(), *result.tolist()]
        rows.append("\t".join(map(str, values)))
        sample = result
Path(sys.argv[1]).write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
