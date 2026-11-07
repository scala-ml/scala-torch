"""Export NUI Python LoRA block-weight plans for both checkpoint families."""
import json
import sys
from pathlib import Path

nui, output = map(Path, sys.argv[1:3])
sys.path.insert(0, str(nui / "python"))
from sdfx.lora_blocks import parse_weights, diffusers_scales
from diffusers.loaders.unet_loader_utils import _maybe_expand_lora_scales_for_one_adapter

specs=['NONE','ALL','ALL0.5','INS','IND','INALL','MIDD','OUTD','OUTS','OUTALL',
       ','.join(str(i/10) for i in range(26)), ','.join(str(i/10) for i in range(20)),
       '0.1,-0.25,0.5','mixed']
cases=[]
for sdxl in (False,True):
 for spec in specs:
  weights=parse_weights(spec,sdxl,{'mixed':'0.7,0.2,-0.4'})
  scales=diffusers_scales(weights,sdxl,.7,.4)
  def vector(v): return v if isinstance(v,list) else [v]
  blocks={'down':[1,2] if sdxl else [0,1,2], 'up':[0,1] if sdxl else [1,2,3]}
  state={'mid_block.attentions.0.to_q.weight':None}
  for side,indices in blocks.items():
   for i in indices:
    for j in range(2 if side=='down' else 3): state[f'{side}_blocks.{i}.attentions.{j}.to_q.weight']=None
  expanded=_maybe_expand_lora_scales_for_one_adapter(scales['unet'],blocks,{'down':2,'up':3},state)
  targets=['mid_block.attentions.0.to_q','mid_block.resnets.0.conv1','conv_in','time_embedding.linear_1',
           'down_blocks.0.resnets.0.conv1','down_blocks.0.resnets.1.conv2',
           'down_blocks.0.downsamplers.0.conv','down_blocks.1.attentions.0.to_q',
           'down_blocks.1.resnets.1.conv1','up_blocks.0.resnets.0.conv1',
           'up_blocks.1.attentions.2.to_v','up_blocks.2.upsamplers.0.conv']
  def resolve(name):
   for key,value in expanded.items():
    if key in name:return value
   parts=name.split('.')
   if len(parts)<4:return None
   return expanded.get(f'{parts[0]}.{parts[1]}.attentions.{parts[3]}',1.0)
  cases.append({'sdxl':sdxl,'spec':spec,'weights':weights,'text':scales['text_encoder'],
    'down':[vector(v) for v in scales['unet']['down'].values()], 'mid':scales['unet']['mid'],
    'up':[vector(v) for v in scales['unet']['up'].values()], 'targets':{'unet.'+name:resolve(name) for name in targets}})
output.write_text(json.dumps(cases),encoding='utf-8')
