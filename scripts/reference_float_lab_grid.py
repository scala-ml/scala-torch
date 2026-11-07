"""Generate the OpenCV float RGB-to-LAB interpolation grid, little-endian u16."""
import sys
from pathlib import Path
import cv2
import numpy as np

v = np.arange(33, dtype=np.float32) / 32
r, g, b = np.meshgrid(v, v, v, indexing="ij")
rgb = np.stack((r, g, b), axis=-1).reshape(1, -1, 3)
lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)
encoded = np.rint((lab + [0, 128, 128]) * [16384 / 100, 64, 64]).astype("<u2")
Path(sys.argv[1]).write_bytes(encoded.tobytes())
