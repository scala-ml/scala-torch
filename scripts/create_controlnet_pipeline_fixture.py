"""Create a small factor-8 VAE pipeline for native ControlNet integration tests.

The other pipeline components are copied from an existing local tiny SD fixture.
The output directory must not already exist.
"""
import shutil
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL

source, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=False)
for name in ("text_encoder", "tokenizer", "unet"):
    shutil.copytree(source / name, output / name)
shutil.copy2(source / "euler-demo.json", output / "scheduler_config.json")
torch.manual_seed(31415)
vae = AutoencoderKL(
    in_channels=3, out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(16, 16, 16, 16), layers_per_block=1,
    latent_channels=4, norm_num_groups=8, sample_size=64,
)
vae.save_pretrained(output / "vae", safe_serialization=True)
