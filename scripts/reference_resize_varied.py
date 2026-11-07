"""Generate Pillow reference resizes for mixed enlargement/reduction geometries."""
import json
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(sys.argv[1]) / "python"))
from sdfx.images import resize_image

cases = []
for w, h, tw, th in ((3, 18, 14, 4), (18, 3, 4, 14), (17, 31, 13, 7),
                     (31, 17, 7, 13), (23, 19, 16, 12), (16, 12, 23, 19)):
    image = Image.new("RGB", (w, h))
    image.putdata([((x * 71 + y * 19) % 256, (x * 13 + y * 83) % 256, (x * 97 + y * 41) % 256)
                   for y in range(h) for x in range(w)])
    for mode in range(3):
        result = resize_image(mode, image, tw, th)
        cases.append(dict(w=w, h=h, tw=tw, th=th, mode=mode,
                          rgb=[v for pixel in result.getdata() for v in pixel]))
print(json.dumps(cases))
