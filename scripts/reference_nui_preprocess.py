"""Capture NUI preprocess behavior using optional local Pillow/OpenCV reference tools."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image
import cv2

root=Path(sys.argv[1]);sys.path.insert(0,str(root/"python"))
from sdfx.controlnet import run_preprocessor,standalone_preprocess,preprocess_infotext
w,h=32,24
pixels=[((x*7+y*11)%256,(x*3+y*17)%256,(x*13+y*5)%256) for y in range(h) for x in range(w)]
image=Image.new("RGB",(w,h));image.putdata(pixels)
cases=[]
for name,args in [("none",{}),("invert",{}),("recolor_luminance",{"gamma_correction":1.3}),
                  ("recolor_intensity",{}),("tile_resample",{"down_sampling_rate":2.0}),
                  ("tile_colorfix",{"variation":3.0}),("blur_gaussian",{"sigma":1.5}),
                  ("canny",{"low_threshold":80.0,"high_threshold":160.0})]:
  for standalone,res,pixel in [(False,-1,False),(True,64,False),(True,-1,True)]:
    if standalone:
      out,info,notes=standalone_preprocess(name,image,args,res,pixel)
      text=preprocess_infotext(info,out.size)
    else:
      out=run_preprocessor(name,image,args);info=None;notes=[];text=None
    cases.append(dict(name=name,args=args,standalone=standalone,processor_res=res,pixel_perfect=pixel,
       width=out.width,height=out.height,pixels=[list(p) for p in out.getdata()],info=info,notes=notes,parameters=text))
path=Path("models/src/test/resources/nui-preprocess-reference.json")
path.write_text(json.dumps(dict(source=hashlib.sha256((root/"python/sdfx/controlnet.py").read_bytes()).hexdigest(),
    opencv=cv2.__version__,width=w,height=h,pixels=pixels,cases=cases),indent=2)+"\n",encoding="utf-8",newline="\n")
print(path)
