"""Optional NUI parser oracle; never imported by the native runtime."""
import argparse
import importlib.util
import json
import hashlib
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--nui", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("nui_prompt", args.nui / "python/sdfx/prompt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
texts = ["", "plain text", "(red [apple])", r"\(red\) \[apple\] \\",
         "(unclosed [apple", "a BREAK b", "BREAK BREAK", "(BREAK)",
         "(apple:2.5)", "(apple:x2.5)", "(apple:x-0.5)", "(apple:x0)",
         "(a (b):x2.2)", "(a:..)", "(a:x..)", "a:1.2) ]", "a BREAK",
         "aBREAKb", "(red:-2) [green]", "word " * 100,
         "a" * 68 + "," + "b" * 25, "a" * 75 + "," + "b" * 20]
randomizer = random.Random(42)
atoms = ["a", " ", "(", ")", "[", "]", ":2)", ":-0.3)", ":x2.5)",
         ":..)", ":x..)", "BREAK", r"\(", r"\\", ",", "日本語"]
texts += ["".join(randomizer.choices(atoms, k=20)) for _ in range(100)]
cases = []
for text in texts:
    spans = module.parse_prompt_attention(text)
    # A deterministic code-point tokenizer isolates parser/chunker behavior from BPE.
    chunks = module.chunk_tokens(spans, lambda ts: [[ord(c) for c in t] for t in ts], 1, 2, 44)
    cases.append(dict(text=text, spans=[[t, str(w)] for t, w in spans],
                      chunks=[dict(tokens=module.apply_pad(t, 2, 0), weights=[str(w) for w in wts])
                              for t, wts in chunks]))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("[\n" + ",\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n]\n",
                       encoding="utf-8", newline="\n")
manifest = dict(source="stablediffusion-nui/python/sdfx/prompt.py",
                source_sha256=hashlib.sha256((args.nui / "python/sdfx/prompt.py").read_bytes()).hexdigest(),
                fixture_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                cases=len(cases), tokenizer="Unicode code points; BOS=1 EOS=2 PAD=0 comma=44",
                chunk_length=75, comma_backtrack=20, random_seed=42)
args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
