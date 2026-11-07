"""Create NUI/Diffusers KL Optimal fixtures; Python is not used for inference."""
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
for xl in (False,True):
 for n in (1,4,11,20):
  sched,_=schedulers.build_scheduler(schedulers.plan('Euler','KL Optimal'),base.config,sdxl=xl)
  sched.set_timesteps(n);refs[f'{xl}:{n}']={'sigmas':sched.sigmas.tolist(),'timesteps':sched.timesteps.tolist()}
(out/'kl-schedule-reference.json').write_text(json.dumps(refs))
scheduler,_=schedulers.build_scheduler(schedulers.plan('Euler','KL Optimal'),base.config)
pipe=StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77),unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=scheduler,safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
with torch.no_grad():
 latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
 result=pipe('a red cat',height=64,width=64,num_inference_steps=4,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(out/'kl-euler-reference.png')

from PIL import Image
image=Image.new('RGB',(64,64))
image.putdata([((x*4)%256,(y*4)%256,64) for y in range(64) for x in range(64)])
image.save(out/'kl-img2img-source.png')
imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
imgpipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('Euler','KL Optimal'),base.config)
with torch.no_grad():
 result=imgpipe('a red cat',image=image,strength=.75,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(out/'kl-img2img-reference.png')
