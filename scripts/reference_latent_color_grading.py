"""NUI latent color-grading vectors and Diffusers full trajectory references."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from diffusers import (AutoencoderKL,EulerDiscreteScheduler,DPMSolverMultistepScheduler,
                       StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,UNet2DConditionModel)
from transformers import CLIPTextModel,CLIPTokenizer
root,nui,output=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx.sampling_ext import LatentCallbacks
torch.set_num_threads(1)
config=dict(recenter=True,recenter_strength=.8,r=.4,g=-.3,b=.7,normalize=True,normalize_strength=.6)
vectors={}
for xl in (False,True):
    for final in (False,True):
        lat=torch.sin(torch.arange(64,dtype=torch.float32)*.17).reshape(1,4,4,4)*6
        cb=LatentCallbacks(xl,4,42);cb.color_grading(config)
        vectors[f'{xl}_{final}']=cb(SimpleNamespace(_num_timesteps=4),3 if final else 1,None,dict(latents=lat))['latents'].flatten().tolist()
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
        for mode,runner in [('txt2img',pipe),('img2img',imgpipe)]:
            runner.scheduler=cls.from_config(scheduler.config)
            cb=LatentCallbacks(False,4,42)
            cb.mirroring(dict(style='90 Degree Rotation',mode='Blend Average',max_step=1.,strength=.6))
            cb.resharpen(dict(sharpness=1.5,scaling='Cos'));cb.color_grading(config)
            args=dict(prompt='a red cat',num_inference_steps=4,guidance_scale=7.5,
                generator=torch.Generator(device='cpu').manual_seed(42),output_type='latent',callback_on_step_end=cb)
            if mode=='img2img':args.update(image=source,strength=.75)
            else:args.update(height=16,width=16,latents=torch.randn((1,4,8,8),generator=torch.Generator(device='cpu').manual_seed(42)))
            cases[f'{mode}_{name}']=runner(**args).images.flatten().tolist()
output.write_text(json.dumps(dict(vectors=vectors,cases=cases)),encoding='utf-8')
