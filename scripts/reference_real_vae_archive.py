"""Create a real VAE archive fixture from an existing SD checkpoint (test only)."""
import sys
import torch
from safetensors import safe_open

source, output = sys.argv[1:3]
with safe_open(source, framework="pt", device="cpu") as checkpoint:
    weights = {key[len("first_stage_model."):]: checkpoint.get_tensor(key).clone()
               for key in checkpoint.keys() if key.startswith("first_stage_model.")}
if not weights:
    raise ValueError("checkpoint has no VAE weights")
key = "decoder.conv_out.bias"
if key not in weights:
    raise ValueError(f"checkpoint has no {key}")
weights[key] = weights[key] + 1
torch.save({"state_dict": weights}, output)
print(f"wrote {len(weights)} VAE tensors, modifying {key}")
