"""Tiny SD NUI CFG Schedule hook + Diffusers reference trajectories."""
import json
import sys
from pathlib import Path
import torch
from diffusers import (AutoencoderKL, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

root,nui,output=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx.sampling_ext import CfgScheduleHook
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
    for name,cls in [('euler',EulerDiscreteScheduler),('ancestral',EulerAncestralDiscreteScheduler),('dpm',DPMSolverMultistepScheduler)]:
        for mode,runner in [('txt2img',pipe),('img2img',imgpipe)]:
            for curve in ['linear','cosine','square','sqrt']:
                runner.scheduler=cls.from_config(scheduler.config)
                config=dict(start_cfg=10.,end_cfg=2.,start=.2,end=.8,curve=curve)
                hook=CfgScheduleHook(runner.unet,runner.scheduler,7.5,config,lambda:True).install()
                try:
                    args=dict(prompt='a red cat',num_inference_steps=4,guidance_scale=7.5,guidance_rescale=.7,
                              generator=torch.Generator(device='cpu').manual_seed(42),output_type='latent')
                    if mode=='img2img': args.update(image=source,strength=.75)
                    else: args.update(height=16,width=16,latents=torch.randn((1,4,8,8),generator=torch.Generator(device='cpu').manual_seed(42)))
                    cases[f'{mode}_{name}_{curve}']=runner(**args).images.flatten().tolist()
                finally: hook.remove()
output.write_text(json.dumps(cases),encoding='utf-8')
