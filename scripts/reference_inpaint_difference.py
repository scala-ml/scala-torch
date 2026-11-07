"""NUI Inpaint Difference binary-mask fixtures, test only."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

nui, output = map(Path, sys.argv[1:3])
sys.path.insert(0, str(nui / "python"))
from sdfx.inpaint_difference import compute

width, height = 11, 9
base = np.empty((height, width, 3), dtype=np.uint8)
altered = np.empty_like(base)
for y in range(height):
    for x in range(width):
        base[y, x] = ((x * 31 + y * 11) % 256, (x * 7 + y * 23) % 256, (x * 13 + y * 5) % 256)
        altered[y, x] = base[y, x]
        if (x * 7 + y * 3) % 13 < 4:
            altered[y, x] = (255 - base[y, x]).astype(np.uint8)
cases = []
for threshold in (0., .25, .5, .99, 1.):
    for dilation, erosion, contours in ((0, 0, False), (1, 0, False), (0, 1, False), (2, 1, False), (0, 0, True), (1, 0, True), (0, 1, True), (2, 1, True), (100, 0, False), (0, 100, False), (100, 100, True)):
        mask, _ = compute(Image.fromarray(base), Image.fromarray(altered), threshold=threshold,
                          dilation=dilation, erosion=erosion, contours=contours)
        cases.append(dict(threshold=threshold, dilation=dilation, erosion=erosion, contours=contours, pixels=list(mask.getdata())))
small = base[::2, ::3].copy()
resized_cases = []
for threshold in (.25, .5, .99, 1.):
    mask, _ = compute(Image.fromarray(small), Image.fromarray(altered), threshold=threshold)
    resized_cases.append(dict(threshold=threshold, pixels=list(mask.getdata())))
output.write_text(json.dumps(dict(width=width, height=height, base=base.flatten().tolist(),
                                  altered=altered.flatten().tolist(), cases=cases,
                                  small_width=small.shape[1], small_height=small.shape[0], small=small.flatten().tolist(),
                                  resized_cases=resized_cases)), encoding="utf-8")
