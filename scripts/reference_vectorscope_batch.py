"""NUI-style batched tiny SD generation with Vectorscope random noise."""
import json,sys
from pathlib import Path
import torch
from diffusers import (AutoencoderKL,EulerDiscreteScheduler,EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler,
                       StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,UNet2DConditionModel)
from transformers import CLIPTextModel,CLIPTokenizer
root,nui,output=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx.sampling_ext import LatentCallbacks
from sdfx.params import hires_steps
torch.set_num_threads(1)
tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77)
scheduler=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',timestep_spacing='linspace')
pipe=StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=tokenizer,
    unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=scheduler,
    safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
cases={}
imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
with torch.no_grad():
 for name,cls in [('euler',EulerDiscreteScheduler),('ancestral',EulerAncestralDiscreteScheduler),('dpm',DPMSolverMultistepScheduler)]:
  for method in ['N.Random','U.Random']:
   pipe.scheduler=cls.from_config(scheduler.config)
   cb=LatentCallbacks(False,4,42)
   cb.vectorscope(dict(brightness=.4,contrast=-.7,saturation=.8,r=.2,g=-.3,b=.6,method=method,scaling='Cos',strength=1.3))
   initial=torch.cat([torch.randn((1,4,8,8),generator=torch.Generator(device='cpu').manual_seed(s)) for s in (42,43)])
   result=pipe(['a red cat']*2,height=16,width=16,num_inference_steps=4,guidance_scale=7.5,
       generator=[torch.Generator(device='cpu').manual_seed(s) for s in (42,43)],latents=initial,
       callback_on_step_end=cb,output_type='latent').images
   cases[f'{name}|{method}']=result.flatten().tolist()
   if name=='euler':
    up=torch.nn.functional.interpolate(result,size=(16,16),mode='nearest-exact')
    count,strength,_=hires_steps(3,.75)
    imgpipe.scheduler=cls.from_config(scheduler.config)
    second=LatentCallbacks(False,count,43)
    second.vectorscope(dict(brightness=.4,contrast=-.7,saturation=.8,r=.2,g=-.3,b=.6,method=method,scaling='Cos',strength=1.3))
    output_latent=imgpipe(['a red cat']*2,image=up,strength=strength,num_inference_steps=count,guidance_scale=7.5,
      generator=[torch.Generator(device='cpu').manual_seed(s) for s in (42,43)],callback_on_step_end=second,output_type='latent').images
    cases[f'hires|{method}']=output_latent.flatten().tolist()
output.write_text(json.dumps(cases),encoding='utf-8')
