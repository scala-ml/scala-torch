"""Create a PyTorch-format test copy of trained safetensors weights, without modifying the source."""
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file

source, destination = map(Path, sys.argv[1:])
if destination.exists():
    raise FileExistsError(destination)
weights = load_file(source, device="cpu")
torch.save({"state_dict": weights, "epoch": 1, "learning_rate": 0.0001}, destination)
print(f"saved {len(weights)} tensors to {destination}")
