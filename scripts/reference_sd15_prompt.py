"""Create pinned single-CLIP prompt tensors using NUI's current Python oracle."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import transformers
from transformers import CLIPTextModel, CLIPTokenizerFast

root, nui, output = map(Path, sys.argv[1:4])
source = nui / "python/sdfx/prompt.py"
spec = importlib.util.spec_from_file_location("nui_prompt", source)
prompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt)
torch.set_num_threads(2)
encoder = CLIPTextModel.from_pretrained(root / "text_encoder", local_files_only=True).eval()
tokenizer = CLIPTokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True)
pad_tokenizer = CLIPTokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True, pad_token="!")
output.mkdir(parents=True, exist_ok=True)

cases = [
    ("plain", "a red apple", "blur", 1, False, False),
    ("empty-negative", "a red apple", "", 1, False, False),
    ("skip2", "a red apple", "blur", 2, False, False),
    ("skip3-plain", "a red apple", "blur", 3, False, False),
    ("weighted-skip2", r"(red:1.5) [apple] \(table\)", "(blur:0.5)", 2, False, False),
    ("skip3-weighted", r"(red:1.5) [apple] \(table\)", "(blur:0.5)", 3, False, False),
    ("long-positive", "apple, " * 40, "blur", 1, False, False),
    ("long-negative", "apple", "blur BREAK noise BREAK fog", 1, False, False),
    ("truncated", "apple, " * 40, "blur", 1, True, False),
    ("custom-pad", "apple, " * 40, "blur", 1, False, True),
]
manifest = {
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "model_revision": "hf-internal-testing/tiny-stable-diffusion-pipe@39f7e1819d772fbbd978486616c6dae068519e85",
    "inputs": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("text_encoder", "tokenizer")
        for path in (root / folder).rglob("*")
        if path.is_file() and path.suffix in {".json", ".txt", ".safetensors"}
    },
    "cases": [],
}

with torch.no_grad():
    for name, positive, negative, clip_skip, truncate, custom_pad in cases:
        active_tokenizer = pad_tokenizer if custom_pad else tokenizer
        pipe = SimpleNamespace(text_encoder=encoder, tokenizer=active_tokenizer)
        pos = prompt.encode_sd15(pipe, positive, clip_skip)
        neg = prompt.encode_sd15(pipe, negative, clip_skip)
        if truncate:
            pos = pos[:, :77]
            neg = neg[:, :77]
        empty = prompt.encode_sd15(pipe, "", clip_skip)
        chunks = max(pos.shape[1], neg.shape[1]) // 77
        if pos.shape[1] < chunks * 77:
            pos = torch.cat([pos, empty.repeat(1, chunks - pos.shape[1] // 77, 1)], dim=1)
        if neg.shape[1] < chunks * 77:
            neg = torch.cat([neg, empty.repeat(1, chunks - neg.shape[1] // 77, 1)], dim=1)
        entry = {"name": name, "prompt": positive, "negative": negative, "clip_skip": clip_skip,
                 "truncate": truncate, "custom_pad": custom_pad, "shapes": {}, "files": {}}
        for label, tensor in (("positive", pos), ("negative", neg)):
            data = tensor.contiguous().cpu().numpy().astype("<f4").tobytes()
            filename = f"{name}-{label}.bin"
            (output / filename).write_bytes(data)
            entry["shapes"][label] = list(tensor.shape)
            entry["files"][label] = {"name": filename, "sha256": hashlib.sha256(data).hexdigest()}
        manifest["cases"].append(entry)

(output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Saved {len(cases)} SD1.5 prompt oracle cases to {output}")
