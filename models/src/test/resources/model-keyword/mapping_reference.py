"""Static CSV/filename matching oracles from NUI and Python stdlib; not a test dependency."""
import difflib
import importlib.util
import json
from pathlib import Path
import random
import sys

spec = importlib.util.spec_from_file_location('mapping_oracle', Path(sys.argv[1]) / 'python/sdfx/model_keyword.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
rng = random.Random(731)
pairs = [('', ''), ('', 'a'), ('tide', 'diet'), ('diet', 'tide'),
         ('ab', 'acab'), ('a' * 199, 'b' + 'a' * 199),
         ('a' * 199, 'b' + 'a' * 198), ('b' + 'a' * 300, 'c' + 'a' * 300),
         ('a' * 300 + 'x', 'a' * 300 + 'y'), ('猫😀.safetensors', '犬😀.safetensors')]
for _ in range(150):
    length = rng.choice([5, 30, 199, 200, 250])
    left = ''.join(rng.choice('abc._猫😀') for _ in range(length))
    right = left[:length // 2] + ''.join(rng.choice('abc._犬😀') for _ in range(length // 2))
    pairs.append((left, right))
records = [dict(left=a, right=b, ratio=difflib.SequenceMatcher(None, a, b).ratio()) for a, b in pairs]
Path(__file__).with_name('similarity.jsonl').write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf8', newline='\n')
texts = ['# comment\nAB, "red, blue|green",model.safetensors\n',
         'a,"line1\r\nline2",file\r\nb,"a ""quote""",file\r',
         'a,,\nshort,words\none\n', 'a,"unterminated', 'a,"b"c,name',
         '  \n\r\n a, spaced , name \n', '\u00a0AB\u0085,\u2003words\u2003,name',
         '"#quoted comment",ignored\na,"",\n', 'a,words,name,extra', 'a,"b""c",name']
csv = [dict(text=text, entries=[vars(entry) for entry in module.parse_rows(text, 1)]) for text in texts]
Path(__file__).with_name('csv.jsonl').write_text(
    ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in csv), encoding='utf8', newline='\n')
print(f'{len(records)} similarity and {len(csv)} CSV cases')
