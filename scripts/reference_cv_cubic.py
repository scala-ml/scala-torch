"""Generate OpenCV cubic-resize fixtures for the native ControlNet preprocessor."""
import json
from pathlib import Path
import cv2
import numpy as np

cases=[]
for w,h,ow,oh,pattern in [(32,24,85,64,"gradient"),(7,5,19,13,"impulse"),
                          (11,9,6,5,"gradient"),(3,2,8,7,"impulse"),(17,13,31,23,"gradient"),
                          (4,4,9,11,"impulse"),(1,5,7,13,"gradient"),(17,3,31,9,"impulse"),
                          (9,12,18,24,"gradient"),(16,11,7,19,"impulse"),(5,7,3,4,"gradient")]:
    pixels=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
      for x in range(w):
        pixels[y,x]=((x*7+y*11)%256,(x*3+y*17)%256,(x*13+y*5)%256) if pattern=="gradient" else (
            (255,80,17) if (x+y)%4==0 else (0,200,255) if x==w-1 or y==h-1 else (15,50,110))
    out=cv2.resize(pixels,(ow,oh),interpolation=cv2.INTER_CUBIC)
    cases.append(dict(w=w,h=h,ow=ow,oh=oh,source=pixels.reshape(-1,3).tolist(),pixels=out.reshape(-1,3).tolist()))
path=Path("models/src/test/resources/cv-cubic-reference.json")
path.write_text(json.dumps(dict(opencv=cv2.__version__,cases=cases),separators=(",",":"))+"\n",encoding="utf-8",newline="\n")
print(path)
