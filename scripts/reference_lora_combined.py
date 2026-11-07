"""Reference combined dynamic rank, block weight and step window behavior."""
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

root,nui,output=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx.lora_blocks import active,parse_weights,diffusers_scales
from sdfx.dynamic_lora import truncate_state_dict
torch.set_num_threads(1)
tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77)
scheduler=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',timestep_spacing='linspace')
pipe=StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),
 text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=tokenizer,
 unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=scheduler,
 safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
layer=pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1.to_q
down=torch.stack([torch.sin(torch.arange(layer.in_features)*.31)*.2,torch.cos(torch.arange(layer.in_features)*.17)*.1])
up=torch.stack([torch.cos(torch.arange(layer.out_features)*.23)*30,torch.sin(torch.arange(layer.out_features)*.11)*15],dim=1)
state={'x.lora_down.weight':down,'x.lora_up.weight':up,'x.alpha':torch.tensor(2.)}
cases={}
with torch.no_grad():
 for spec,rank,start,stop in [('ALL',2,None,None),('NONE',2,None,None),('MIDD',1,2,None),('ALL',1,None,2),('ALL',0,2,None)]:
  pipe.scheduler=EulerDiscreteScheduler.from_config(scheduler.config)
  transformed=truncate_state_dict(state,rank)
  a=transformed['x.lora_down.weight'];b=transformed['x.lora_up.weight'];alpha=transformed['x.alpha']/a.shape[0]
  scale=diffusers_scales(parse_weights(spec,False),False,1.,1.)['unet']['mid']
  current={'step':1};trace=[]
  def hook(_module,args,result):
   return result+F.linear(F.linear(args[0],a),b)*(alpha*scale) if active(current['step'],4,start,stop) else result
  def callback(_pipe,i,_time,values):
   trace.append(values['latents'].flatten().tolist());current['step']=i+2;return values
  h=layer.register_forward_hook(hook)
  try:
   pipe(prompt='a red cat',height=16,width=16,num_inference_steps=4,guidance_scale=7.5,
    latents=torch.randn((1,4,8,8),generator=torch.Generator().manual_seed(42)),
    generator=torch.Generator().manual_seed(42),output_type='latent',callback_on_step_end=callback)
   cases[f'{spec}|{rank}|{start}|{stop}']=trace
  finally:h.remove()
output.write_text(json.dumps(cases),encoding='utf-8')
