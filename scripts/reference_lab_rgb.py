"""Generate deterministic OpenCV LAB-to-RGB reference pixels."""
import cv2
import numpy as np
from pathlib import Path
import sys

rng = np.random.default_rng(912)
lab = rng.integers(0, 256, (1, 65536, 3), dtype=np.uint8)
lab[0, :5] = [[0, 128, 128], [255, 128, 128], [128, 0, 255], [128, 255, 0], [255, 255, 255]]
rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
Path(sys.argv[1]).write_bytes(lab.tobytes() + rgb.tobytes())
