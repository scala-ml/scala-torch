"""Create a synthetic nine-channel tiny SD fixture; never used by the worker."""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

source, destination = map(Path, sys.argv[1:3])
shutil.copytree(source, destination)
weights_path = destination / "unet/diffusion_pytorch_model.safetensors"
weights = load_file(weights_path)
old = weights["conv_in.weight"]
wide = torch.zeros((old.shape[0], 9, *old.shape[2:]), dtype=old.dtype)
wide[:, :4] = old
for channel in range(4, 9):
    wide[:, channel].fill_(.025 * (channel - 3))
weights["conv_in.weight"] = wide
save_file(weights, weights_path)
config_path = destination / "unet/config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["in_channels"] = 9
config_path.write_text(json.dumps(config), encoding="utf-8")
