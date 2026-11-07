"""Generate NUI/Pillow inpaint fill reference values; not used at runtime."""
import json
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(sys.argv[1]) / "python"))
from sdfx.inpaint import prepare

w, h = 13, 9
pixels = [((x * 31 + y * 9) % 256, (x * 17 + y * 39) % 256, (x * 3 + y * 73) % 256)
          for y in range(h) for x in range(w)]
source = Image.new("RGB", (w, h))
source.putdata(pixels)
patterns = {
    "edge": [255 if x >= 6 else 0 for y in range(h) for x in range(w)],
    "gradient": [(x * 37 + y * 53 + x * y * 7) % 256 for y in range(h) for x in range(w)],
    "point": [255 if (x, y) == (0, 4) else 0 for y in range(h) for x in range(w)],
}
cases = []
for name, levels in patterns.items():
    mask = Image.new("L", (w, h))
    mask.putdata(levels)
    for blur in (0, 4):
        for fill in (0, 1, 2, 3):
            job = prepare(source, mask, w, h, mask_blur=blur, inpainting_fill=fill, seed=42)
            cases.append(dict(pattern=name, mask=levels, blur=blur, fill=fill,
                              result=[(r << 16) | (g << 8) | b for r, g, b in job.image.getdata()]))
json.dump(dict(width=w, height=h, pixels=[(r << 16) | (g << 8) | b for r, g, b in pixels],
               cases=cases), sys.stdout, separators=(",", ":"))
