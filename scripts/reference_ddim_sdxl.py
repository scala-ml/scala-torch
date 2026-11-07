"""Create independent SDXL DDIM references; Python is not used by the native worker."""
from pathlib import Path
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
root,nui,out=map(Path,sys.argv[1:4]);sys.path.insert(0,str(nui/'python'))
from sdfx import schedulers
torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
pipe=StableDiffusionXLPipeline.from_pretrained(root,local_files_only=True,add_watermarker=False).to('cpu')
base=pipe.scheduler.config
# NUI applies its own weighted dual-CLIP encoding; use its checked-in oracle.
embeds=Path('models/src/test/resources/tiny-sdxl-prompt')
def read(name,shape):
 return torch.frombuffer(bytearray((embeds/(name+'.bin')).read_bytes()),dtype=torch.float32).reshape(shape)
conditioning=dict(prompt_embeds=read('plain-positive',(1,77,64)),negative_prompt_embeds=read('plain-negative',(1,77,64)),pooled_prompt_embeds=read('plain-pooled',(1,32)),negative_pooled_prompt_embeds=read('plain-negative-pooled',(1,32)))
image=Image.new('RGB',(64,64));image.putdata([((x*4)%256,(y*4)%256,64) for y in range(64) for x in range(64)]);image.save(out/'ddim-sdxl-source.png')
for eta in (0.0,1.0):
 pipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base,sdxl=True,extra_config={'clip_sample':False,'steps_offset':0})
 with torch.no_grad():
  latent=torch.randn((1,4,32,32),generator=torch.Generator().manual_seed(42))
  result=pipe(**conditioning,height=64,width=64,num_inference_steps=4,guidance_scale=7.5,latents=latent,generator=torch.Generator().manual_seed(42),eta=eta,output_type='pil').images[0]
 result.save(out/f'ddim-sdxl-txt-eta{int(eta)}.png')
 imgpipe=StableDiffusionXLImg2ImgPipeline(**pipe.components).to('cpu')
 imgpipe.scheduler,_=schedulers.build_scheduler(schedulers.plan('DDIM','Uniform'),base,sdxl=True,extra_config={'clip_sample':False,'steps_offset':0})
 with torch.no_grad():
  result=imgpipe(**conditioning,image=image,strength=.75,num_inference_steps=4,guidance_scale=7.5,generator=torch.Generator().manual_seed(42),eta=eta,output_type='pil').images[0]
 result.save(out/f'ddim-sdxl-img-eta{int(eta)}.png')
