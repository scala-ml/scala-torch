"""Optional Pillow/NUI fixture generation; not required by Scala tests/runtime."""
import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(sys.argv[1])/'python'))
from sdfx.images import load_rgb
out=Path('models/src/test/resources/image-input')
out.mkdir(parents=True,exist_ok=True)
im=Image.new('RGBA',(3,2))
im.putdata([(240,30,10,255),(10,220,30,128),(30,10,240,0),(150,80,10,64),(70,10,130,200),(90,190,10,1)])
for orientation in range(1,9):
    exif=Image.Exif();exif[274]=orientation
    name=f'orientation-{orientation}.png'
    im.save(out/name,exif=exif)
    (out/(name+'.rgb')).write_bytes(load_rgb(str(out/name)).tobytes())
    jpeg=f'orientation-{orientation}.jpg'
    im.convert('RGB').save(out/jpeg,quality=100,subsampling=0,exif=exif)
    (out/(jpeg+'.rgb')).write_bytes(load_rgb(str(out/jpeg)).tobytes())
    tiff=f'orientation-{orientation}.tiff'
    im.convert('RGB').save(out/tiff,exif=exif)
    (out/(tiff+'.rgb')).write_bytes(load_rgb(str(out/tiff)).tobytes())
for mode in ['L','LA','P']:
    image=Image.new(mode,(3,2))
    if mode=='P':
        image.putpalette([v for i in range(256) for v in (i,255-i,(i*7)%256)])
        image.putdata([0,50,100,150,200,255]);image.info['transparency']=bytes([i for i in range(256)])
    elif mode=='L': image.putdata([0,50,100,150,200,255])
    else: image.putdata([(0,255),(50,128),(100,0),(150,64),(200,200),(255,1)])
    name=f'mode-{mode}.png';image.save(out/name)
    (out/(name+'.rgb')).write_bytes(load_rgb(str(out/name)).tobytes())
