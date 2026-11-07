"""Export one trained LoRA layer and Python NUI dynamic-rank reference values."""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

source, nui, fixture, reference = map(Path, sys.argv[1:5])
sys.path.insert(0, str(nui / "python"))
from sdfx.dynamic_lora import truncate_state_dict

base = "lora_unet_down_blocks_0_attentions_0_proj_in"
suffixes = (".lora_down.weight", ".lora_up.weight", ".alpha")
with safe_open(source, framework="pt", device="cpu") as archive:
    state = {base + suffix: archive.get_tensor(base + suffix).float() for suffix in suffixes}
fixture.parent.mkdir(parents=True, exist_ok=True)
save_file(state, fixture)
torch.set_num_threads(1)
x = torch.sin(torch.arange(320, dtype=torch.float32) * 0.013).reshape(1, 320, 1, 1)
result = {}
for rank in (0, 1, 8, 64, -1):
    transformed = truncate_state_dict(state, rank)
    down = transformed[base + suffixes[0]]
    up = transformed[base + suffixes[1]]
    alpha = transformed[base + suffixes[2]]
    output = F.conv2d(F.conv2d(x, down), up) * (alpha / down.shape[0])
    result[str(rank)] = output.flatten().tolist()
reference.write_text(json.dumps(result), encoding="utf-8")
