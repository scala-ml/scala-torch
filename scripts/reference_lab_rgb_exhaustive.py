"""Generate OpenCV RGB results for all 24-bit LAB values."""
import sys
from pathlib import Path
import cv2
import numpy as np

output = Path(sys.argv[1])
with output.open("wb") as file:
    for lightness in range(256):
        a, b = np.indices((256, 256), dtype=np.uint8)
        lab = np.stack((np.full_like(a, lightness), a, b), axis=-1)
        file.write(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).tobytes())
