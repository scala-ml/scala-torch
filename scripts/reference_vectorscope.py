"""Independent NUI Vectorscope callback fixtures for SD1 and SDXL."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(sys.argv[1])/'python'))
from sdfx.sampling_ext import LatentCallbacks
cases={}
for xl in (False,True):
 for method in ('Straight','Cross','N.Random','U.Random'):
  for scaling in ('Flat','Cos','Sin','1 - Cos','1 - Sin'):
   cb=LatentCallbacks(xl,4,42)
   cb.vectorscope(dict(brightness=.4,contrast=-.7,saturation=.8,r=.2,g=-.3,b=.6,method=method,scaling=scaling,strength=1.3))
   pipe=SimpleNamespace(_num_timesteps=4)
   lat=torch.sin(torch.arange(64,dtype=torch.float32)*.17).reshape(1,4,4,4)
   for i in range(4):lat=cb(pipe,i,None,dict(latents=lat))['latents']
   cases[f'{xl}|{method}|{scaling}']=lat.flatten().tolist()
for seed in (42,43):
 cb=LatentCallbacks(False,4,seed)
 cb.vectorscope(dict(brightness=.4,contrast=-.7,saturation=.8,r=.2,g=-.3,b=.6,method='N.Random',scaling='Cos',strength=1.3))
 lat=torch.sin(torch.arange(128,dtype=torch.float32)*.17).reshape(2,4,4,4)
 for i in range(4):lat=cb(SimpleNamespace(_num_timesteps=4),i,None,dict(latents=lat))['latents']
 cases[f'batch|{seed}']=lat.flatten().tolist()
Path(sys.argv[2]).write_text(json.dumps(cases),encoding='utf-8')
