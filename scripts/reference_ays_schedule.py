"""Create NUI/Diffusers Align Your Steps fixtures; Python is not used for inference."""
import json,sys
from pathlib import Path
import torch
from diffusers import AutoencoderKL,EulerDiscreteScheduler,StableDiffusionPipeline,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
root,nui,out=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx import schedulers
torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
base=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear')
refs={}
for xl in (False,True):
 for n in (1,4,11,20):
  sched,_=schedulers.build_scheduler(schedulers.plan('Euler','Align Your Steps'),base.config,sdxl=xl)
  sched.set_timesteps(n);refs[f'{xl}:{n}']={'sigmas':sched.sigmas.tolist(),'timesteps':sched.timesteps.tolist()}
(out/'ays-schedule-reference.json').write_text(json.dumps(refs))
scheduler,_=schedulers.build_scheduler(schedulers.plan('Euler','Align Your Steps'),base.config)
pipe=StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77),unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=scheduler,safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
with torch.no_grad():
 latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
 result=pipe('a red cat',height=64,width=64,num_inference_steps=4,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(out/'ays-euler-reference.png')
