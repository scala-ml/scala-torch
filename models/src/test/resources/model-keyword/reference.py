"""Regenerate static keyword fixtures from an existing NUI checkout; never run by tests."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys

source = Path(sys.argv[1]) / 'python/sdfx/model_keyword.py'
spec = importlib.util.spec_from_file_location('keyword_oracle', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
records = []
for index, (seed, choice, lora_choice, placement, order) in enumerate(itertools.product(
        [42, -9223372036854775808], ['keyword1, keyword2', 'random', 'iterate'],
        ['None', '< random >', '< iterate >'],
        ['keyword prompt', 'prompt keyword', 'keyword, prompt', 'prompt, keyword'],
        ['textual inversion first', 'model keyword first'])):
    model = [None, 'red| blue |green', '', 'one'][index % 4]
    lora = ['x|y|z', None, 'single'][index % 3]
    embedding = [None, 'ti', '', 'None'][index % 4]
    lora_model = None if index % 5 == 0 else 'nested/style.safetensors'
    weight = [0.7, 0.00001, 0.0001, -0.0][index % 4]
    options = dict(model_choice=choice, lora_choice=lora_choice, placement=placement,
                   order=order, embedding=embedding, lora_model=lora_model, multiplier=repr(weight))
    prompts = ['one', 'two']
    preview, expected = module.plan_with_preview('preview', prompts, model, choice,
        lora, lora_choice, None if embedding in ('', 'None') else embedding,
        order, placement, lora_model, weight, 3, 2, random.Random(seed))
    state = module.IterationState()
    state._model, state._lora = 3, 2
    state.reserve(len(prompts), model, choice, lora, lora_choice)
    records.append(dict(seed=str(seed), model=model, lora=lora, options=options,
        preview=preview, prompts=expected, next_model=state._model, next_lora=state._lora))
destination = Path(__file__).with_name('reference.jsonl')
destination.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records), encoding='utf8', newline='\n')
print(f'{len(records)} cases; source SHA256 {hashlib.sha256(source.read_bytes()).hexdigest()}')
