"""Reference tiny SD trajectories with a windowed U-Net-only linear LoRA."""
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

root, nui, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(nui / "python"))
from sdfx.lora_blocks import active

torch.set_num_threads(1)
tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77)
scheduler=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',timestep_spacing='linspace')
pipe=StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=tokenizer,
    unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=scheduler,
    safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
layer=pipe.unet.time_embedding.linear_1
down=torch.full((1,layer.in_features),1/layer.in_features,dtype=torch.float32)
up=torch.full((layer.out_features,1),8.,dtype=torch.float32)
source=((torch.arange(3*16*16).reshape(1,3,16,16)%256).float()/127.5)-1
results={}
with torch.no_grad():
 for kind,runner in [('txt2img',pipe),('img2img',imgpipe)]:
  for sampler,cls in [('euler',EulerDiscreteScheduler),('ancestral',EulerAncestralDiscreteScheduler),('dpm',DPMSolverMultistepScheduler)]:
   for start,stop in [(None,None),(2,None),(None,2),(.5,.9),(5,None)]:
    runner.scheduler=cls.from_config(scheduler.config)
    total=4 if kind=='txt2img' else 3
    state={'step':1};trace=[]
    def apply(_module,args,result):
     return result+F.linear(F.linear(args[0],down),up) if active(state['step'],total,start,stop) else result
    def callback(_pipe,index,_time,values):
     trace.append(values['latents'].flatten().tolist())
     state['step']=index+2
     return values
    handle=layer.register_forward_hook(apply)
    try:
     args=dict(prompt='a red cat',num_inference_steps=4,guidance_scale=7.5,
       generator=torch.Generator(device='cpu').manual_seed(42),output_type='latent',callback_on_step_end=callback)
     if kind=='img2img':args.update(image=source,strength=.75)
     else:args.update(height=16,width=16,latents=torch.randn((1,4,8,8),generator=torch.Generator(device='cpu').manual_seed(42)))
     runner(**args)
     results[f'{kind}|{sampler}|{start}|{stop}']=trace
    finally: handle.remove()
output.write_text(json.dumps(results),encoding='utf-8')
