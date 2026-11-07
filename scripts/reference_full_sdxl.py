"""Create opt-in full SDXL FP32 comparison fixtures; never used by the runtime.

Usage: python scripts/reference_full_sdxl.py MODEL_ROOT NUI_ROOT OUTPUT_DIR
Requires the independent PyTorch, Diffusers, Transformers and tokenizer packages.
The model root must contain the standard unvariant safetensors file names.
"""
import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizerFast

root, nui, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

def save(name, value):
    (out / f"{name}.bin").write_bytes(value.detach().cpu().float().contiguous().numpy().astype("<f4").tobytes())

spec = importlib.util.spec_from_file_location("nui_prompt", nui / "python/sdfx/prompt.py")
prompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt)
pipe = SimpleNamespace(
    text_encoder=CLIPTextModel.from_pretrained(root / "text_encoder", local_files_only=True, torch_dtype=torch.float32).float().eval(),
    text_encoder_2=CLIPTextModelWithProjection.from_pretrained(root / "text_encoder_2", local_files_only=True, torch_dtype=torch.float32).float().eval(),
    tokenizer=CLIPTokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True),
    tokenizer_2=CLIPTokenizerFast.from_pretrained(root / "tokenizer_2", local_files_only=True))
with torch.no_grad():
    positive, pooled = prompt.encode_sdxl(pipe, "a photograph of a red apple on a wooden table", 1)
    save("prompt-positive", positive)
    save("prompt-pooled", pooled)
del pipe
gc.collect()

with torch.no_grad():
    sample = torch.sin(torch.arange(256, dtype=torch.float32) * .13).reshape(1, 4, 8, 8)
    context = torch.cos(torch.arange(8 * 2048, dtype=torch.float32) * .07).reshape(1, 8, 2048)
    pooled = torch.sin(torch.arange(1280, dtype=torch.float32) * .11).reshape(1, 1280)
    ids = torch.tensor([[256., 256., 0., 0., 256., 256.]])
    for name, value in [("sample", sample), ("context", context), ("pooled", pooled), ("ids", ids)]:
        save(name, value)
    vae = AutoencoderKL.from_pretrained(root / "vae", local_files_only=True, torch_dtype=torch.float32).float().eval()
    save("vae-output", vae.decode(sample).sample)
    del vae
    gc.collect()
    unet = UNet2DConditionModel.from_pretrained(root / "unet", local_files_only=True, torch_dtype=torch.float32).float().eval()
    save("unet-output", unet(sample, 500, encoder_hidden_states=context,
         added_cond_kwargs={"text_embeds": pooled, "time_ids": ids}).sample)
print(f"Full SDXL FP32 reference fixtures saved to {out}")
