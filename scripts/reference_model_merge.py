"""Create independent NUI model merge references for the native worker."""
from pathlib import Path
import sys,torch
from safetensors.torch import save_file
root,out=map(Path,sys.argv[1:3]);sys.path.insert(0,str(root/'python'));out.mkdir(parents=True,exist_ok=True)
from sdfx import model_tools
keys=['model.diffusion_model.input_blocks.0.conv.weight','model.diffusion_model.middle_block.conv.weight','first_stage_model.conv.weight','conditioner.test.weight']
def create(name,shift,channels,missing=False):
 state={keys[0]:torch.arange(8*channels,dtype=torch.float32).reshape(2,channels,2,2)+shift,
        keys[1]:torch.tensor([1.2,-3.4],dtype=torch.float16)+shift,
        keys[2]:torch.tensor([2.0,8.0],dtype=torch.float32)+shift,
        keys[3]:torch.tensor([.3,-.7],dtype=torch.float64)+shift,
        'integer':torch.tensor([123456,-7],dtype=torch.int32)}
 if missing: state['only_a']=torch.tensor([1.5],dtype=torch.float32)
 save_file(state,out/(name+'.safetensors'))
create('a',0,4,True);create('b',1,2);create('c',2,4)
for mode in model_tools.MODES:
 for precision in ('fp16','bf16','fp32'):
  name=mode.lower().replace(' ','-')+'-'+precision
  model_tools.merge(str(out/'a.safetensors'),str(out/'b.safetensors'),str(out/(name+'.safetensors')),
   c=str(out/'c.safetensors'),mode=mode,alpha=.35,beta=.2,block_weights='preset:GRAD_V',vae_source='B',precision=precision)
