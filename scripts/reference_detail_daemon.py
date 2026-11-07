"""NUI Detail Daemon envelopes and sigma/timestep transformations."""
import json,sys
from pathlib import Path
import torch
from diffusers import EulerDiscreteScheduler
sys.path.insert(0,str(Path(sys.argv[1])/'python'))
from sdfx.sampling_ext import make_dd_schedule,SigmaTable
s=EulerDiscreteScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear')
tab=SigmaTable(s.alphas_cumprod)
cases=[]
for steps in [1,2,5,20]:
 for start,end,bias,exponent,smooth in [(.2,.8,.5,1,True),(.8,.2,.5,1,True),(0,1,.0,0,False),(.1,.9,.8,2,False)]:
  opts=dict(start=start,end=end,bias=bias,amount=.4,exponent=exponent,start_offset=-.1,end_offset=.2,fade=.15,smooth=smooth)
  values=make_dd_schedule(steps,**opts).tolist()
  cases.append(dict(steps=steps,options=opts,values=values))
transforms=[]
for t in [0.,.25,100.,499.5,999.]:
 for m in [-.5,.1,.8]:
  sig=tab.sigma(t);s2=min(max(sig*max(1-m,1e-3),float(tab.sig[0])),float(tab.sig[-1]))
  transforms.append(dict(t=t,m=m,scale=((1+sig*sig)/(1+s2*s2))**.5,target=tab.t(s2)))
Path(sys.argv[2]).write_text(json.dumps(dict(cases=cases,transforms=transforms)),encoding='utf-8')
