"""Optional NUI dual-CLIP oracle, using local tiny SDXL and fast tokenizers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch
import transformers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizerFast

root, nui = map(Path, sys.argv[1:3])
source = nui / "python/sdfx/prompt.py"
spec = importlib.util.spec_from_file_location("nui_prompt", source)
prompt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt)
torch.set_num_threads(2)
pipe = SimpleNamespace(text_encoder=CLIPTextModel.from_pretrained(root / "text_encoder", local_files_only=True).eval(),
                       text_encoder_2=CLIPTextModelWithProjection.from_pretrained(root / "text_encoder_2", local_files_only=True).eval(),
                       tokenizer=CLIPTokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True),
                       tokenizer_2=CLIPTokenizerFast.from_pretrained(root / "tokenizer_2", local_files_only=True))
out = Path("models/src/test/resources/tiny-sdxl-prompt")
out.mkdir(parents=True, exist_ok=True)
manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), torch=torch.__version__,
                transformers=transformers.__version__, tokenizer="CLIPTokenizerFast", cases=[],
                model_revision="fa06b5c3d6d45fabd978486616c6dae068519e85",
                inputs={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for part in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
                        for p in (root / part).iterdir() if p.suffix in [".json", ".txt", ".safetensors"]})
original_tokenizer2 = pipe.tokenizer_2
temporary = tempfile.TemporaryDirectory()
empty_merges = Path(temporary.name) / "merges.txt"
empty_merges.write_text("#version: 0.2\n", encoding="utf-8")
split_tokenizer2 = CLIPTokenizerFast(vocab_file=str(root / "tokenizer_2/vocab.json"), merges_file=str(empty_merges))
pad_tokenizer2 = CLIPTokenizerFast.from_pretrained(root / "tokenizer_2", local_files_only=True, pad_token="!")
cases = [("plain", "a red apple", "blur", 1), ("skip2", "a red apple", "blur", 2),
         ("skip3", "a red apple", "blur", 3), ("empty", "a red apple", "", 1),
         ("whitespace", "a red apple", " \t\n\u3000\u0085", 1),
         ("weighted", r"(red:1.5) [apple] \(table\)", "(blur:0.5)", 1),
         ("long", "apple, " * 40, "blur", 1),
         ("negative-long", "apple", "blur BREAK noise BREAK fog", 1),
         ("repeat", "(red:x2.5) BREAK apple", "blur", 3),
         ("split", "apple, " * 25, "blur", 1), ("pad", "a red apple", "blur", 1)]
with torch.no_grad():
    for name, text, negative, skip in cases:
        pipe.tokenizer_2 = split_tokenizer2 if name == "split" else pad_tokenizer2 if name == "pad" else original_tokenizer2
        pe, pp = prompt.encode_sdxl(pipe, text, skip)
        if not negative.strip():
            ne, np = torch.zeros_like(pe), torch.zeros_like(pp)
        else:
            ne, np = prompt.encode_sdxl(pipe, negative, skip)
            if pe.shape[1] != ne.shape[1]:
                empty, _ = prompt.encode_sdxl(pipe, "", skip)
                pe, ne = prompt.pad_to_match(pe, ne, empty)
        case = dict(name=name, prompt=text, negative=negative, clip_skip=skip, tensors={})
        case["encoder_chunks"] = [len(prompt._chunks_for(pipe, tok, text)) for tok in [pipe.tokenizer, pipe.tokenizer_2]]
        for label, tensor in [("positive", pe), ("negative", ne), ("pooled", pp), ("negative-pooled", np)]:
            data = tensor.contiguous().numpy().astype("<f4").tobytes()
            filename = f"{name}-{label}.bin"
            (out / filename).write_bytes(data)
            case["tensors"][label] = dict(file=filename, shape=list(tensor.shape), sha256=hashlib.sha256(data).hexdigest())
        manifest["cases"].append(case)
(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"Saved {len(cases)} native-conditioning reference cases")
temporary.cleanup()
