"""Generate deterministic OpenCV RGB-to-LAB reference pixels."""
import cv2
import numpy as np
from pathlib import Path
import sys

rng = np.random.default_rng(463)
rgb = rng.integers(0, 256, (1, 65536, 3), dtype=np.uint8)
rgb[0, :8] = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0],
              [0, 0, 255], [128, 128, 128], [1, 1, 1], [254, 254, 254]]
lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
Path(sys.argv[1]).write_bytes(rgb.tobytes() + lab.tobytes())
