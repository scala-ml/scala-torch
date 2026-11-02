"""Optional CPU oracle preparation for pinned tiny SDXL; never a native runtime dependency.

Usage: python scripts/reference_sdxl.py <downloaded-tiny-sdxl-root>
Verifies upstream pickle weights before weights_only loading and creates local
safetensors plus independent component fixtures. No model code is downloaded.
"""
import hashlib
import json
from pathlib import Path
import sys

import torch
import diffusers
import transformers
from safetensors.torch import save_file
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection

root = Path(sys.argv[1])
revision = "fa06b5c3d6d45fabd978486616c6dae068519e85"
metadata = json.loads((root / "source-metadata.json").read_text(encoding="utf-8-sig"))
assert metadata["sha"] == revision
sources = {entry["rfilename"]: entry for entry in metadata["siblings"]}
for part in ["text_encoder", "text_encoder_2", "unet", "vae"]:
    name = "pytorch_model.bin" if part.startswith("text") else "diffusion_pytorch_model.bin"
    path = root / part / name
    expected = sources[f"{part}/{name}"]["lfs"]["sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    weights = torch.load(path, map_location="cpu", weights_only=True)
    target = "model.safetensors" if part.startswith("text") else "diffusion_pytorch_model.safetensors"
    save_file({key: value.contiguous() for key, value in weights.items()}, root / part / target)

torch.set_num_threads(2)
out = Path("models/src/test/resources/tiny-sdxl")
out.mkdir(parents=True, exist_ok=True)
manifest = {"source": "hf-internal-testing/tiny-stable-diffusion-xl-pipe", "revision": revision,
            "torch": torch.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "tensors": {}}

def save(name, tensor):
    data = tensor.detach().cpu().float().contiguous().numpy().astype("<f4").tobytes()
    (out / f"{name}.bin").write_bytes(data)
    manifest["tensors"][name] = {"shape": list(tensor.shape), "sha256": hashlib.sha256(data).hexdigest()}

with torch.no_grad():
    ids = torch.tensor([[0, 17, 22, 2, 1, 1], [0, 29, 2, 1, 1, 1]])
    manifest["token_ids"] = ids.tolist()
    for part, cls in [("text_encoder", CLIPTextModel), ("text_encoder_2", CLIPTextModelWithProjection)]:
        model = cls.from_pretrained(root / part, local_files_only=True).eval()
        result = model(ids, output_hidden_states=True)
        save(f"{part}-penultimate", result.hidden_states[-2])
        if part == "text_encoder_2":
            save(f"{part}-projected", result.text_embeds)
        else:
            save(f"{part}-final", result.last_hidden_state)
    unet = UNet2DConditionModel.from_pretrained(root / "unet", local_files_only=True).eval()
    sample = torch.sin(torch.arange(256, dtype=torch.float32) * .13).reshape(1, 4, 8, 8)
    context = torch.cos(torch.arange(6*64, dtype=torch.float32) * .07).reshape(1, 6, 64)
    pooled = torch.sin(torch.arange(32, dtype=torch.float32) * .11).reshape(1, 32)
    time_ids = torch.tensor([[64., 80., 3., 5., 64., 64.]])
    for name, tensor in [("sample", sample), ("context", context), ("pooled", pooled), ("time-ids", time_ids)]:
        save(name, tensor)
    result = unet(sample, 500, encoder_hidden_states=context,
                  added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
    save("unet-output", result)
    sample2 = torch.cat([sample, -sample], dim=0)
    context2 = torch.cat([context, context * .3], dim=0)
    pooled2 = torch.cat([pooled, -pooled], dim=0)
    time_ids2 = torch.tensor([[64., 80., 3., 5., 64., 64.], [96., 128., 7., 9., 64., 80.]])
    for name, tensor in [("sample-b2", sample2), ("context-b2", context2), ("pooled-b2", pooled2), ("time-ids-b2", time_ids2)]:
        save(name, tensor)
    save("unet-output-b2", unet(sample2, 500, encoder_hidden_states=context2,
         added_cond_kwargs={"text_embeds": pooled2, "time_ids": time_ids2}).sample)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Prepared {len(manifest['tensors'])} independent SDXL tensors at {out}")
