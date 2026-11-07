from pathlib import Path
import json,torch
from diffusers import DDIMScheduler
out={}
for spacing in ('linspace','leading','trailing'):
 for clip in (False,True):
  for pred in ('epsilon','sample','v_prediction'):
   p=DDIMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',timestep_spacing=spacing,steps_offset=1,set_alpha_to_one=False,clip_sample=clip,prediction_type=pred)
   p.set_timesteps(4)
   sample=torch.tensor([[-2.5,-.5,.5,2.5]],dtype=torch.float32);model=torch.tensor([[.1,-.4,.3,-.2]],dtype=torch.float32)
   values=[]
   for t in p.timesteps:
    v=p.step(model,t,sample).prev_sample
    values.append(v.flatten().tolist())
   out[f'{spacing}:{clip}:{pred}']={'timesteps':p.timesteps.tolist(),'values':values}
   eta_values=[]
   for t in p.timesteps:
    v=p.step(model,t,sample,eta=-.5).prev_sample
    eta_values.append(v.flatten().tolist())
   out[f'{spacing}:{clip}:{pred}']['negative_eta']=eta_values
Path('models/src/test/resources/ddim-sampler-reference.json').write_text(json.dumps(out))

# End-to-end fixtures, using NUI's scheduler construction.
import sys
from diffusers import AutoencoderKL,EulerDiscreteScheduler,StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
from PIL import Image
root,nui,outdir=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'));outdir.mkdir(parents=True,exist_ok=True)
from sdfx import schedulers
torch.set_num_threads(1)
base=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear')
pipe=StableDiffusionPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77),unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=base,safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
source=Image.new('RGB',(64,64));source.putdata([((x*4)%256,(y*4)%256,64) for y in range(64) for x in range(64)]);source.save(outdir/'ddim-sampler-source.png')
for clip in (False,True):
 for schedule in ('Uniform','DDIM','Simple'):
  pipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM',schedule),base.config,extra_config={'clip_sample':clip,'steps_offset':0})
  latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
  with torch.no_grad():
   result=pipe('a red cat',height=64,width=64,num_inference_steps=4,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
  result.save(outdir/f'ddim-sampler-{schedule}-{clip}.png')
 imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
 imgpipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':clip,'steps_offset':0})
 with torch.no_grad():
  result=imgpipe('a red cat',image=source,strength=.75,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(outdir/f'ddim-sampler-img2img-{clip}.png')

pipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','DDIM'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
trace=[]
def capture(p,step,t,values):
 trace.append(values['latents'].detach().cpu().contiguous().numpy().astype('<f4').tobytes());return values
with torch.no_grad():
 latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
 pipe('a red cat',height=64,width=64,num_inference_steps=4,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),output_type='pil',callback_on_step_end=capture)
(outdir/'ddim-sampler-trace.f32').write_bytes(b''.join(trace))

# Full-mask inpainting references for four- and nine-channel U-Nets.
from diffusers import StableDiffusionInpaintPipeline
nine=Path(sys.argv[4])
for channels,modelroot in ((4,root),(9,nine)):
 inpaint=StableDiffusionInpaintPipeline(vae=AutoencoderKL.from_pretrained(modelroot/'vae',local_files_only=True),text_encoder=CLIPTextModel.from_pretrained(modelroot/'text_encoder',local_files_only=True),tokenizer=CLIPTokenizer(vocab_file=str(modelroot/'tokenizer/vocab.json'),merges_file=str(modelroot/'tokenizer/merges.txt'),model_max_length=77),unet=UNet2DConditionModel.from_pretrained(modelroot/'unet',local_files_only=True),scheduler=base,safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
 old=inpaint.prepare_mask_latents
 def masked(mask,image,*args,**kwargs):
  return old(mask,image.contiguous(),*args,**kwargs)
 inpaint.prepare_mask_latents=masked
 inpaint.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
 with torch.no_grad():
  result=inpaint('a red cat',image=source,mask_image=Image.new('L',(64,64),255),height=64,width=64,strength=.75,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(outdir/f'ddim-sampler-inpaint-{channels}.png')

# Stochastic DDIM with eta=1, seeded CPU generator.
pipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
with torch.no_grad():
 latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
 result=pipe('a red cat',height=64,width=64,num_inference_steps=4,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),eta=1.0,output_type='pil').images[0]
 result.save(outdir/'ddim-sampler-eta-txt2img.png')
imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
imgpipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
with torch.no_grad():
 result=imgpipe('a red cat',image=source,strength=.75,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),eta=1.0,output_type='pil').images[0]
 result.save(outdir/'ddim-sampler-eta-img2img.png')

# Pixel Hires 4-channel and decoded latent Hires 9-channel.
import torch.nn.functional as F
pipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
with torch.no_grad():
 latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
 first=pipe('a red cat',height=64,width=64,num_inference_steps=2,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 imgpipe=StableDiffusionImg2ImgPipeline(**pipe.components).to('cpu')
 imgpipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
 result=imgpipe('a red cat',image=first.resize((128,128),Image.Resampling.NEAREST),strength=.5,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(outdir/'ddim-sampler-pixel-hires.png')
 inpaint.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
 first=inpaint('a red cat',image=Image.new('RGB',(64,64),(127,127,127)),mask_image=Image.new('L',(64,64),255),height=64,width=64,strength=1.,num_inference_steps=2,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='latent').images
 scaled=F.interpolate(first.float(),size=(16,16),mode='nearest')
 decoded=inpaint.vae.decode(scaled/inpaint.vae.config.scaling_factor,return_dict=False)[0]
 image=inpaint.image_processor.postprocess(decoded,output_type='pil')[0]
 inpaint.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
 result=inpaint('a red cat',image=image,mask_image=Image.new('L',(128,128),255),height=128,width=128,strength=.5,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 result.save(outdir/'ddim-sampler-nine-latent-hires.png')

# Half-mask inpainting: NUI composites preserved pixels after diffusion.
mask=Image.new('L',(64,64));mask.putdata([255 if x>=32 else 0 for y in range(64) for x in range(64)])
mask.save(outdir/'ddim-sampler-half-mask.png')
for channels,modelroot in ((4,root),(9,nine)):
 inpaint=StableDiffusionInpaintPipeline(vae=AutoencoderKL.from_pretrained(modelroot/'vae',local_files_only=True),text_encoder=CLIPTextModel.from_pretrained(modelroot/'text_encoder',local_files_only=True),tokenizer=CLIPTokenizer(vocab_file=str(modelroot/'tokenizer/vocab.json'),merges_file=str(modelroot/'tokenizer/merges.txt'),model_max_length=77),unet=UNet2DConditionModel.from_pretrained(modelroot/'unet',local_files_only=True),scheduler=base,safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
 old=inpaint.prepare_mask_latents
 def masked(mask,image,*args,**kwargs):
  return old(mask,image.contiguous(),*args,**kwargs)
 inpaint.prepare_mask_latents=masked
 for eta in (0.0,1.0):
  inpaint.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base.config,extra_config={'clip_sample':False,'steps_offset':0})
  with torch.no_grad():
   result=inpaint('a red cat',image=source,mask_image=mask,height=64,width=64,strength=.75,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),eta=eta,output_type='pil').images[0]
  Image.composite(result,source,mask).save(outdir/f'ddim-sampler-half-{channels}-eta{int(eta)}.png')
