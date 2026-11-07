"""NUI Detail Daemon U-Net hook on Diffusers tiny SD trajectories."""
import json,sys
from pathlib import Path
import torch
from diffusers import (AutoencoderKL,EulerDiscreteScheduler,DPMSolverMultistepScheduler,
    StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,UNet2DConditionModel)
from transformers import CLIPTextModel,CLIPTokenizer
root,nui,output=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx.sampling_ext import DetailDaemonHook
torch.set_num_threads(1)
tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77)
scheduler=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',timestep_spacing='linspace')
pipe=StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=tokenizer,
    unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=scheduler,
    safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
source=((torch.arange(3*16*16).reshape(1,3,16,16)%256).float()/127.5)-1
cases={}
with torch.no_grad():
 for name,cls in [('euler',EulerDiscreteScheduler),('dpm',DPMSolverMultistepScheduler)]:
  for kind,runner in [('txt2img',pipe),('img2img',imgpipe)]:
   for mode in ['both','cond','uncond']:
    for cfg in (1.,7.5):
     runner.scheduler=cls.from_config(scheduler.config)
     options=dict(mode=mode,amount=.35,start=.2,end=.8,bias=.5,exponent=1.,start_offset=.1,end_offset=.15,fade=0.,smooth=True)
     hook=DetailDaemonHook(runner.unet,runner.scheduler,runner.scheduler.alphas_cumprod,options,lambda:cfg>1).install()
     try:
      args=dict(prompt='a red cat',num_inference_steps=4,guidance_scale=cfg,
          generator=torch.Generator(device='cpu').manual_seed(42),output_type='latent')
      if kind=='img2img':args.update(image=source,strength=.75)
      else:args.update(height=16,width=16,latents=torch.randn((1,4,8,8),generator=torch.Generator(device='cpu').manual_seed(42)))
      cases[f'{kind}|{name}|{mode}|{cfg}']=runner(**args).images.flatten().tolist()
     finally:hook.remove()
output.write_text(json.dumps(cases),encoding='utf-8')
