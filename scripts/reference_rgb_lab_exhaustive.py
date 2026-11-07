"""Generate all 24-bit RGB colors' OpenCV LAB values for an opt-in test."""
import sys
from pathlib import Path
import cv2
import numpy as np

output = Path(sys.argv[1])
with output.open("wb") as file:
    for red in range(256):
        green, blue = np.indices((256, 256), dtype=np.uint8)
        rgb = np.stack((np.full_like(green, red), green, blue), axis=-1)
        file.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).tobytes())
