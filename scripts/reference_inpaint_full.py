"""Generate NUI/Pillow full-resolution inpainting preprocessing references."""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(sys.argv[1]) / "python"))
from sdfx.inpaint import prepare

width, height = 23, 19
source = Image.new("RGB", (width, height))
source.putdata([((x * 23 + y * 7) % 256, (x * 11 + y * 31) % 256, (x * 37 + y * 13) % 256)
                for y in range(height) for x in range(width)])
generated = Image.new("RGB", (16, 12))
generated.putdata([((x * 19 + y * 11) % 256, (x * 29 + y * 3) % 256, (x * 5 + y * 37) % 256)
                   for y in range(12) for x in range(16)])
cases = []
for name, box in (("center", (8, 6, 13, 11)), ("edge", (0, 0, 5, 8)),
                  ("wide", (3, 8, 20, 10)), ("empty", (0, 0, 0, 0))):
    for blur in (0, 2):
        for padding in (0, 3):
            mask = Image.new("L", (width, height))
            mask.putdata([255 if box[0] <= x < box[2] and box[1] <= y < box[3] else 0
                          for y in range(height) for x in range(width)])
            job = prepare(source, mask, 16, 12, mask_blur=blur, full_res=True, full_res_padding=padding)
            cases.append(dict(name=name, blur=blur, padding=padding, mask=list(mask.getdata()),
                              box=job.paste_box, model_image=list(job.image.getdata()),
                              model_mask=list(job.mask.getdata()), final=list(job.finalize(generated).getdata())))
print(json.dumps(dict(width=width, height=height, source=list(source.getdata()),
                      generated=list(generated.getdata()), cases=cases)))
