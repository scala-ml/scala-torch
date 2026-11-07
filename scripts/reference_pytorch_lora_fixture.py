"""Generate a PyTorch LoRA fixture; Python is only needed for this reference fixture."""
from collections import OrderedDict
from pathlib import Path
import sys
import torch

torch.save(OrderedDict([
    ("unet.test.lora_down.weight", torch.tensor([[1.0], [2.0]])),
    ("unet.test.lora_up.weight", torch.tensor([[3.0, 4.0]])),
    ("unet.test.alpha", torch.tensor(4.0)),
]), Path(sys.argv[1]))

if len(sys.argv) > 2:
    tiny = OrderedDict([
        ("lora_unet_time_embedding_linear_1.lora_down.weight", torch.full((1, 32), 1 / 32)),
        ("lora_unet_time_embedding_linear_1.lora_up.weight", torch.full((128, 1), 8.0)),
    ])
    torch.save(tiny, Path(sys.argv[2]))
    if len(sys.argv) > 3:
        torch.save({"state_dict": tiny, "epoch": 12, "name": "adapter", "learning_rate": 0.0001}, Path(sys.argv[3]))
