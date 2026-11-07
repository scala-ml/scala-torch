"""Generate NUI A1111 LAB histogram transfer reference stages."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

sys.path.insert(0, str(Path(sys.argv[1]) / "python"))
from sdfx.color_correction import a1111

cases = []
for w, h, rw, rh, salt in ((11, 7, 7, 13, 0), (1, 1, 1, 1, 4),
                            (3, 17, 13, 4, 29), (18, 5, 2, 9, 17)):
    def make(width, height, seed):
        return np.array([[( (x * 53 + y * 29 + seed) % 256,
                            (x * 11 + y * 71 + seed * 3) % 256,
                            (x * 31 + y * 7 + seed * 5) % 256)
                          for x in range(width)] for y in range(height)], dtype=np.uint8)
    source, reference = make(w, h, 2 + salt), make(rw, rh, 17 + salt)
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
    matched = match_histograms(source_lab, reference_lab, channel_axis=2)
    cases.append(dict(width=w, height=h, ref_width=rw, ref_height=rh,
                      source_lab=source_lab.ravel().tolist(), reference_lab=reference_lab.ravel().tolist(),
                      matched=matched.ravel().tolist(),
                      final=np.asarray(a1111(Image.fromarray(source), Image.fromarray(reference))).ravel().tolist()))
print(json.dumps(cases))
