"""Create NUI/Diffusers DDIM fixtures; Python is not used for inference."""
import json,sys
from pathlib import Path
import torch
from diffusers import AutoencoderKL,EulerDiscreteScheduler,StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
root,nui,out=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx import schedulers
torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
base=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear')
refs={}
for sampler in ('Euler','Euler a','DPM++ 2M'):
 for offset in (0,1):
  for n in (1,4,7,11,20):
   sched,_=schedulers.build_scheduler(schedulers.plan(sampler,'DDIM'),base.config,extra_config={'steps_offset':offset})
   sched.set_timesteps(n);refs[f'{sampler}:{offset}:{n}']={'sigmas':sched.sigmas.tolist(),'timesteps':sched.timesteps.tolist()}
(out/'ddim-schedule-reference.json').write_text(json.dumps(refs))
scheduler,_=schedulers.build_scheduler(schedulers.plan('Euler','DDIM'),base.config)
pipe=StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77),unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=scheduler,safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
with torch.no_grad():
 latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
 result=pipe('a red cat',height=64,width=64,num_inference_steps=4,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(out/'ddim-euler-reference.png')

for sampler,filename in [('Euler a','ddim-ancestral-reference.png'),('DPM++ 2M','ddim-dpm-reference.png')]:
 pipe.scheduler,_=schedulers.build_scheduler(schedulers.plan(sampler,'DDIM'),base.config)
 with torch.no_grad():
  latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
  result=pipe('a red cat',height=64,width=64,num_inference_steps=4,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
  result.save(out/filename)

from PIL import Image
image=Image.new('RGB',(64,64))
image.putdata([((x*4)%256,(y*4)%256,64) for y in range(64) for x in range(64)])
image.save(out/'ddim-img2img-source.png')
imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
imgpipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('Euler','DDIM'),base.config)
trace=[]
def callback(p,step,t,values):
 trace.append(values['latents'].detach().cpu().contiguous().numpy().astype('<f4').tobytes());return values
with torch.no_grad():
 result=imgpipe('a red cat',image=image,strength=.6,num_inference_steps=7,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil',callback_on_step_end=callback).images[0]
 result.save(out/'ddim-img2img-reference.png')
 (out/'ddim-img2img-latents.f32').write_bytes(b''.join(trace))
print('actual img2img steps:',len(trace))

pipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('Euler','DDIM'),base.config)
with torch.no_grad():
 latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
 result=pipe('a red cat',height=64,width=64,num_inference_steps=7,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(out/'ddim-seven-reference.png')
