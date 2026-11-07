"""Generate independent NUI color correction pixel fixtures."""
import json
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(sys.argv[1]) / "python"))
from sdfx.color_correction import a1111, adain, wavelet

def make(w, h, salt):
    image = Image.new("RGB", (w, h))
    image.putdata([((x * 53 + y * 29 + salt) % 256, (x * 11 + y * 71 + salt * 3) % 256,
                    (x * 31 + y * 7 + salt * 5) % 256) for y in range(h) for x in range(w)])
    return image

source, reference = make(11, 7, 2), make(7, 13, 17)
cases = []
for method, fn in (("A1111", a1111), ("AdaIN", adain), ("Wavelet", wavelet)):
    for strength in (0, .4, 1):
        result = fn(source, reference, strength)
        cases.append(dict(method=method, strength=strength,
                          rgb=[v for p in result.getdata() for v in p]))
print(json.dumps(cases))
