"""Generate NUI color-enhance reference data for native image-tool tests."""
import json
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(sys.argv[1]) / "python"))
from sdfx.image_tools import color_enhance

cases = []
for kind in ("color", "gray"):
    image = Image.new("RGB", (19, 17))
    image.putdata([(((x * 31 + y * 17) % 256,) * 3 if kind == "gray" else
                    ((x * 23 + y * 7) % 256, (x * 11 + y * 19) % 256, (x * 3 + y * 29) % 256))
                   for y in range(17) for x in range(19)])
    for strength in (0, .5, 1):
        for percentile in (70, 99.5):
            output, info = color_enhance(image, strength, percentile)
            cases.append(dict(kind=kind, strength=strength, percentile=percentile, info=info,
                              rgb=[v for p in output.getdata() for v in p]))
print(json.dumps(cases))
