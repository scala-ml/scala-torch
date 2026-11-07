"""Capture NUI inspection results using local Transformers tokenizers, not used at runtime."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from transformers import CLIPTokenizer

nui=Path(sys.argv[1]);models=Path(sys.argv[2]);sys.path.insert(0,str(nui / "python"))
from sdfx.tokenizer_tool import inspect
cases=[]
for profile in ("tiny-sd-pipe","tiny-sdxl-pipe"):
    root=models/profile
    tok=CLIPTokenizer(vocab_file=str(root/"tokenizer/vocab.json"),merges_file=str(root/"tokenizer/merges.txt"))
    assert tok.fix_text is None, "fixture requires Transformers without optional ftfy"
    pipe=SimpleNamespace(tokenizer=tok)
    if profile=="tiny-sdxl-pipe":
        pipe.tokenizer_2=CLIPTokenizer(vocab_file=str(root/"tokenizer_2/vocab.json"),merges_file=str(root/"tokenizer_2/merges.txt"))
    for mode,text in [("text","Hello, world!"),("text","猫 ☕ café"),("text","日本語 中国語"),
                      ("text","A\u0000B\u200bC\ufffdD"),("text","ΟΣ όΣ ABC"),("text","ΟΣ\u0000Α"),
                      ("text","a\u000bb"),("text","&amp; test"),("text",""),
                      ("text","<|startoftext|> cat <|endoftext|>"),("ids","0, 1, 2, 512, 513")]:
        result=inspect(pipe,dict(text=text,input_mode=mode))
        for stream in result["streams"]:stream["source"]=""
        cases.append(dict(profile=profile,mode=mode,text=text,result=result))
path=Path("models/src/test/resources/nui-tokenize-reference.json")
path.write_text(json.dumps(dict(source=hashlib.sha256((nui/"python/sdfx/tokenizer_tool.py").read_bytes()).hexdigest(),
    cases=cases),ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
print(path)
