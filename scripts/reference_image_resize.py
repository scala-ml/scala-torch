import sys, json
from pathlib import Path
from PIL import Image
# Test fixture generation only; normal Scala builds/runtime do not use Python.
# Usage: python scripts/reference_image_resize.py /path/to/stablediffusion-nui
sys.path.insert(0, str(Path(sys.argv[1]) / 'python'))
from sdfx.images import resize_image
out=Path('models/src/test/resources/image-resize')
out.mkdir(parents=True,exist_ok=True)
cases=[]
for w,h,tw,th in [(7,5,13,11),(13,11,5,3),(9,4,6,9),(4,9,9,6),(1,7,4,3),(7,1,3,4),(7,5,7,5)]:
    image=Image.new('RGB',(w,h))
    image.putdata([((x*71+y*19)%256,(x*13+y*83)%256,(x*97+y*41)%256) for y in range(h) for x in range(w)])
    for mode in range(3):
        name=f'{w}-{h}-{tw}-{th}-{mode}.rgb'
        out.joinpath(name).write_bytes(resize_image(mode,image,tw,th).tobytes())
        cases.append([w,h,tw,th,mode,name])
out.joinpath('cases.json').write_text(json.dumps(cases),encoding='utf8')
