"""Produce deterministic NUI CAS reference vectors; not used by the worker."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
from sdfx.image_tools import cas_upscale

pixels = [(x * 61 + y * 13, y * 79 + x * 7, (x * 37 + y * 91) % 256,
           (x * 83 + y * 41) % 256) for y in range(3) for x in range(4)]
image = Image.new("RGBA", (4, 3))
image.putdata(pixels)
cases = []
for scale, sharpness in [(1, 0), (1, 1), (1.5, .5), (2, 1), (.5, .25)]:
    result, info = cas_upscale(image, scale, sharpness)
    cases.append(dict(scale=scale, sharpness=sharpness, width=result.width, height=result.height,
                      pixels=[list(p) for p in result.getdata()], info=info))
path = Path("models/src/test/resources/nui-cas-reference.json")
path.write_text(json.dumps(dict(source_sha256=hashlib.sha256((root / "python/sdfx/image_tools.py").read_bytes()).hexdigest(),
    width=4, height=3, pixels=pixels, cases=cases), indent=2) + "\n", encoding="utf-8", newline="\n")
print(path)
