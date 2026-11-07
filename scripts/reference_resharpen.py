"""Reference NUI ReSharpen with Diffusers txt2img and img2img samplers."""
import json, sys
from pathlib import Path
import torch
from diffusers import (AutoencoderKL, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer

root,nui,output=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx.sampling_ext import LatentCallbacks
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
            for scale in ['Flat','Cos','Sin','1 - Cos','1 - Sin']:
                runner.scheduler=cls.from_config(scheduler.config)
                cb=LatentCallbacks(False,4,42);cb.resharpen(dict(sharpness=1.5,scaling=scale))
                args=dict(prompt='a red cat',num_inference_steps=4,guidance_scale=7.5,
                    generator=torch.Generator(device='cpu').manual_seed(42),output_type='latent',callback_on_step_end=cb)
                if mode=='img2img': args.update(image=source,strength=.75)
                else: args.update(height=16,width=16,latents=torch.randn((1,4,8,8),generator=torch.Generator(device='cpu').manual_seed(42)))
                cases[f'{mode}_{name}_{scale}']=runner(**args).images.flatten().tolist()
    inpaint=StableDiffusionInpaintPipeline(**pipe.components).to('cpu')
    mask=torch.zeros((1,1,16,16),dtype=torch.float32);mask[:,:,:,8:]=1
    for channels in (4,9):
        if channels==9:
            old=inpaint.unet.conv_in
            widened=torch.nn.Conv2d(9,old.out_channels,old.kernel_size,old.stride,old.padding)
            widened.weight[:,:4]=old.weight
            for channel in range(4,9): widened.weight[:,channel].fill_(.025*(channel-3))
            widened.bias.copy_(old.bias);inpaint.unet.conv_in=widened;inpaint.unet.register_to_config(in_channels=9)
        for name,cls in [('euler',EulerDiscreteScheduler),('ancestral',EulerAncestralDiscreteScheduler),('dpm',DPMSolverMultistepScheduler)]:
            for strength in (.75,1.):
                inpaint.scheduler=cls.from_config(scheduler.config)
                cb=LatentCallbacks(False,4,42);cb.resharpen(dict(sharpness=1.5,scaling='Cos'))
                result=inpaint('a red cat',image=source,mask_image=mask,height=16,width=16,strength=strength,
                    num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator(device='cpu').manual_seed(42),
                    output_type='latent',callback_on_step_end=cb).images
                cases[f'inpaint_{channels}_{name}_{int(strength*100)}']=result.flatten().tolist()
output.write_text(json.dumps(cases),encoding='utf-8')
