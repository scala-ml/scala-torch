"""Create non-zero tiny ControlNet and independent conditioned prediction fixtures."""
import json,sys
from pathlib import Path
import torch
from diffusers import ControlNetModel,UNet2DConditionModel,EulerDiscreteScheduler
root,nui,control_dir,output=map(Path,sys.argv[1:5]);sys.path.insert(0,str(nui/'python'))
from sdfx.sampling_ext import DetailDaemonHook
torch.set_num_threads(1);torch.manual_seed(123)
unet=UNet2DConditionModel.from_pretrained(root/'unet',local_files_only=True).eval()
control=ControlNetModel.from_unet(unet,conditioning_embedding_out_channels=(4,8,16,32)).eval()
with torch.no_grad():
 for name,param in control.named_parameters():
  if name.startswith(('controlnet_down_blocks.','controlnet_mid_block.','controlnet_cond_embedding.conv_out.')):
   values=torch.arange(param.numel(),dtype=torch.float32).reshape(param.shape)
   param.copy_(torch.sin(values*.031+len(name))*.012)
 control_dir.mkdir(parents=True,exist_ok=True);control.save_pretrained(control_dir,safe_serialization=True)
 sample=torch.sin(torch.arange(4*8*8,dtype=torch.float32)*.07).reshape(1,4,8,8)
 image=((torch.arange(3*64*64,dtype=torch.float32)%251)/250).reshape(1,3,64,64)
 positive=torch.sin(torch.arange(77*32,dtype=torch.float32)*.013).reshape(1,77,32)
 negative=torch.cos(torch.arange(77*32,dtype=torch.float32)*.017).reshape(1,77,32)
 t=torch.tensor(666.,dtype=torch.float32)
 sched=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear')
 sched.set_timesteps(4)
 # ControlNet executes before the Detail Daemon U-Net forward hook.
 pair=torch.cat([negative,positive]);batch=sample.repeat(2,1,1,1);images=image.repeat(2,1,1,1)
 down,mid=control(batch,t,encoder_hidden_states=pair,controlnet_cond=images,conditioning_scale=.7,return_dict=False)
 result={}
 for mode in ('none','both','cond','uncond'):
  hook=None
  if mode!='none':
   options=dict(mode=mode,amount=.35,start=.2,end=.8,bias=.5,exponent=1.,start_offset=.1,end_offset=.15,fade=0.,smooth=True)
   hook=DetailDaemonHook(unet,sched,sched.alphas_cumprod,options,lambda:True).install()
  try:
   pred=unet(batch,t,encoder_hidden_states=pair,down_block_additional_residuals=down,mid_block_additional_residual=mid).sample
   neg,pos=pred.chunk(2);result[mode]=(neg+7.5*(pos-neg)).flatten().tolist()
  finally:
   if hook:hook.remove()
output.write_text(json.dumps(result),encoding='utf-8')
