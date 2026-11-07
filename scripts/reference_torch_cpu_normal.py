"""Generate CPU randn fixtures; used only as an external test oracle."""
import json
import sys
import torch

cases = []
for seed in (0, 42, 2**32 + 42, 2**63 - 1):
    for sizes in ([1], [15], [16], [17], [31], [32], [33], [64], [4096], [1, 2, 15, 17, 3, 32]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        batches = [torch.randn(n, generator=generator, dtype=torch.float32).tolist() for n in sizes]
        cases.append(dict(seed=str(seed), sizes=sizes, values=batches))
json.dump(dict(torch_version=torch.__version__, cpu=torch.backends.cpu.get_cpu_capability(), cases=cases),
          sys.stdout, separators=(",", ":"))
