"""Verify native SD1/SDXL embeddings with upstream safetensors and PyTorch."""
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file
import sys

root=Path(sys.argv[1])
sd1=load_file(root/"sd1.safetensors")
assert set(sd1)=={"clip_l"}
assert sd1["clip_l"].tolist()==[[1.,2.,3.],[4.,5.,6.]]
with safe_open(root/"sd1.safetensors",framework="pt") as f:
    assert f.metadata()=={"step":"42"}
xl=load_file(root/"sdxl.safetensors")
assert set(xl)=={"clip_l","clip_g"}
assert xl["clip_l"].tolist()==[[1.,2.,3.],[4.,5.,6.]]
assert xl["clip_g"].tolist()==[[7.,8.],[9.,10.]]
print("upstream SD1 and SDXL embeddings verified")
