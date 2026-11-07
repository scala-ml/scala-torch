"""Independent Diffusers/Pillow reference for NUI's two-pass pixel Hires."""
import sys
from pathlib import Path
import torch
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

root, nui, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(nui / 'python'))
from sdfx.params import hires_steps, center_crop_box

torch.set_num_threads(1)
tokenizer = CLIPTokenizer(vocab_file=str(root / 'tokenizer/vocab.json'),
    merges_file=str(root / 'tokenizer/merges.txt'), model_max_length=77)
scheduler = EulerDiscreteScheduler(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
    beta_schedule='scaled_linear', timestep_spacing='linspace')
pipe = StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root / 'vae', local_files_only=True),
    text_encoder=CLIPTextModel.from_pretrained(root / 'text_encoder', local_files_only=True),
    tokenizer=tokenizer, unet=UNet2DConditionModel.from_pretrained(root / 'unet', local_files_only=True),
    scheduler=scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to('cpu')
imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
with torch.no_grad():
    output.mkdir(parents=True, exist_ok=True)
    for sampler, cls, options in [('Euler',EulerDiscreteScheduler,{}),
                                   ('Euler-a',EulerAncestralDiscreteScheduler,{}),
                                   ('DPM-2M',DPMSolverMultistepScheduler,{'use_karras_sigmas':True})]:
        pipe.scheduler=cls.from_config(scheduler.config,**options)
        latent = torch.randn((1, 4, 32, 32), generator=torch.Generator().manual_seed(42))
        base = pipe('a red cat', height=64, width=64, num_inference_steps=2, guidance_scale=7.5,
            latents=latent, generator=torch.Generator().manual_seed(42), output_type='pil').images[0]
        base.save(output / f'{sampler}-base.png')
        for name, method in [('Nearest',Image.Resampling.NEAREST),('Lanczos',Image.Resampling.LANCZOS)]:
            source=base.resize((128,128),method)
            source.save(output / f'{sampler}-{name}-source.png')
            imgpipe.scheduler=cls.from_config(scheduler.config,**options)
            n,strength,_=hires_steps(2,.5)
            trace=[]
            def callback(_pipe, _step, _time, values):
                trace.append(values['latents'].detach().cpu().contiguous().numpy().astype('<f4').tobytes())
                return values
            image=imgpipe('a red cat',image=source,strength=strength,num_inference_steps=n,guidance_scale=7.5,
                generator=torch.Generator().manual_seed(42),output_type='pil',callback_on_step_end=callback).images[0]
            image.save(output / f'{sampler}-{name}-result.png')
            (output / f'{sampler}-{name}-latents.f32').write_bytes(b''.join(trace))
        if sampler == 'Euler':
            for name, method in [('Nearest',Image.Resampling.NEAREST),('Lanczos',Image.Resampling.LANCZOS)]:
                source=base.resize((128,128),method).crop(center_crop_box(128,128,128,72,8))
                imgpipe.scheduler=cls.from_config(scheduler.config,**options)
                n,strength,_=hires_steps(2,.5)
                image=imgpipe('a red cat',image=source,strength=strength,num_inference_steps=n,guidance_scale=7.5,
                    generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
                image.save(output / f'Euler-{name}-crop-result.png')
