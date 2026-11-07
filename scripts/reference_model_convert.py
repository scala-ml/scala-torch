"""Generate small reference checkpoints with the NUI Python implementation."""
from pathlib import Path
import sys
import torch
from safetensors.torch import save_file

root, out = map(Path, sys.argv[1:3])
sys.path.insert(0, str(root / "python"))
from sdfx.model_tools import convert

out.mkdir(parents=True, exist_ok=True)
state = {
    "model.diffusion_model.middle_block.1.weight": torch.tensor([1.25, -2.5]),
    "model_ema.diffusion_modelmiddle_block1weight": torch.tensor([8.5, 9.25]),
    "first_stage_model.decoder.weight": torch.tensor([4.0, 6.0], dtype=torch.float16),
    "cond_stage_model.transformer.position_ids": torch.tensor([[44, 44, 44]], dtype=torch.int64),
    "optimizer.state": torch.tensor([8], dtype=torch.int32),
    "other": torch.tensor([1.0, -0.0], dtype=torch.float64),
}
vae = {"decoder.weight": torch.tensor([7.0, 11.0]), "loss.ignore": torch.tensor([1.0])}
save_file(state, out / "source.safetensors")
save_file(vae, out / "vae.safetensors")
torch.save({"state_dict": vae}, out / "vae.pt")
for precision in ("keep", "fp16", "bf16", "fp32"):
    for ema in ("no-ema", "ema-only", "keep"):
        for fix, drop, bake in ((True, True, True), (False, False, False)):
            name = f"{precision}-{ema}-{int(fix)}"
            convert(str(out / "source.safetensors"), str(out / (name + ".safetensors")),
                    precision=precision, ema=ema, fix_clip=fix, drop_junk=drop,
                    bake_vae=str(out / "vae.safetensors") if bake else None)
