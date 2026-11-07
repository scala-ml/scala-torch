"""Reference for nine-channel pixel Hires using NCHW-contiguous VAE input.

Usage: python reference_nine_pixel_hires.py MODEL_DIR OUTPUT_DIR
Generate MODEL_DIR with reference_nine_channel_fixture.py; Python is used only
for independent test fixtures, not by the native runtime.
"""
import sys
from pathlib import Path
import torch
from PIL import Image
from diffusers import AutoencoderKL,EulerDiscreteScheduler,StableDiffusionInpaintPipeline,UNet2DConditionModel
from transformers import CLIPTextModel,CLIPTokenizer
root,out=map(Path,sys.argv[1:3])
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
pipe=StableDiffusionInpaintPipeline(vae=AutoencoderKL.from_pretrained(root/'vae',local_files_only=True),text_encoder=CLIPTextModel.from_pretrained(root/'text_encoder',local_files_only=True),tokenizer=CLIPTokenizer(vocab_file=str(root/'tokenizer/vocab.json'),merges_file=str(root/'tokenizer/merges.txt'),model_max_length=77),unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True),scheduler=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',timestep_spacing='linspace'),safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cpu')
old=pipe.prepare_mask_latents
def prepare(mask,masked_image,*args,**kwargs):
 return old(mask,masked_image.contiguous(),*args,**kwargs)
pipe.prepare_mask_latents=prepare
trace=[]
def callback(pipe,step,t,values):
 trace.append(values['latents'].detach().cpu().contiguous().numpy().astype('<f4').tobytes());return values
with torch.no_grad():
 base=pipe('a red cat',image=Image.new('RGB',(64,64),(127,127,127)),mask_image=Image.new('L',(64,64),255),height=64,width=64,strength=1.,num_inference_steps=2,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
 base.save(out/'nine-contiguous-base.png')
 source=base.resize((128,128),Image.Resampling.NEAREST)
 source.save(out/'nine-contiguous-source.png')
 result=pipe('a red cat',image=source,mask_image=Image.new('L',(128,128),255),height=128,width=128,strength=.5,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil',callback_on_step_end=callback).images[0]
 result.save(out/'nine-contiguous-output.png');(out/'nine-contiguous-latents.f32').write_bytes(b''.join(trace))

# NUI resizes in latent space at an eight-pixel scale, decodes to PIL, then
# calls the inpaint pipeline. The tiny fixture has a nonstandard VAE scale of 2;
# Diffusers therefore resizes that decoded image during preprocessing.
import torch.nn.functional as F
with torch.no_grad():
 latent=pipe('a red cat',image=Image.new('RGB',(64,64),(127,127,127)),mask_image=Image.new('L',(64,64),255),height=64,width=64,strength=1.,num_inference_steps=2,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='latent').images
 for mode in ('nearest','bilinear'):
  scaled=F.interpolate(latent.float(),size=(16,16),mode=mode,**({'align_corners':False} if mode=='bilinear' else {}))
  decoded=pipe.vae.decode(scaled/pipe.vae.config.scaling_factor,return_dict=False)[0]
  source=pipe.image_processor.postprocess(decoded,output_type='pil')[0]
  source.save(out/f'nine-latent-{mode}-source.png')
  result=pipe('a red cat',image=source,mask_image=Image.new('L',(128,128),255),height=128,width=128,strength=.5,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
  result.save(out/f'nine-latent-{mode}-output.png')

 # The 128x72 crop uses NUI's latent-space centre crop after 128x128 upscale.
 with torch.no_grad():
  scaled=F.interpolate(latent.float(),size=(16,16),mode='nearest')[:,:,3:12,:]
  decoded=pipe.vae.decode(scaled/pipe.vae.config.scaling_factor,return_dict=False)[0]
  source=pipe.image_processor.postprocess(decoded,output_type='pil')[0]
  result=pipe('a red cat',image=source,mask_image=Image.new('L',(128,72),255),height=72,width=128,strength=.5,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),output_type='pil').images[0]
  result.save(out/'nine-latent-nearest-crop-output.png')
