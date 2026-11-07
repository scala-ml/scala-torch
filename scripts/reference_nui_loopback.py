"""Capture NUI Loopback Scaler finishing fixtures; not used by the worker."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

root=Path(sys.argv[1])
sys.path.insert(0,str(root / "python"))
from sdfx.loopback_scaler import run

cases=[]
for width,height in [(7,6),(2,2)]:
    pixels=[((x*41+y*17)%256,(x*9+y*67)%256,(x*33+y*29)%256,(x*31+y*57)%256)
            for y in range(height) for x in range(width)]
    image=Image.new("RGBA",(width,height));image.putdata(pixels)
    for params in [{},{"sharpness":.3},{"sharpness":1.8},{"brightness":1.4},
                   {"color":.2},{"contrast":1.7},{"detail":"Low"},{"detail":"Medium"},{"detail":"High"},{"smooth":"Medium"},
                   {"blur":"Low"},{"contour":True},
                   {"sharpness":1.4,"brightness":.8,"color":1.3,"contrast":1.2,
                    "detail":"Low","smooth":"Low","blur":"Low","contour":True}]:
        result,_=run(image,params)
        cases.append(dict(width=width,height=height,source=pixels,params=params,
                          pixels=[list(p) for p in result.getdata()]))
path=Path("models/src/test/resources/nui-loopback-reference.json")
path.write_text(json.dumps(dict(source_sha256=hashlib.sha256((root / "python/sdfx/loopback_scaler.py").read_bytes()).hexdigest(),
    cases=cases),indent=2)+"\n",encoding="utf-8",newline="\n")
print(path)
