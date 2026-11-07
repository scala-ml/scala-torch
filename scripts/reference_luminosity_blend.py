"""Generate byte fixtures for NUI's luminosity blend stage."""
import json
import numpy as np
from PIL import Image
from blendmodes.blend import blendLayers, BlendType

rng = np.random.default_rng(2947)
cases = []
for count in (1, 16, 257, 1024):
    background = rng.integers(0, 256, (1, count, 3), dtype=np.uint8)
    foreground = rng.integers(0, 256, (1, count, 3), dtype=np.uint8)
    if count >= 16:
        background[0, :4] = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255]]
        foreground[0, :4] = [[255, 255, 255], [0, 0, 0], [0, 255, 0], [255, 0, 0]]
    output = blendLayers(Image.fromarray(background), Image.fromarray(foreground), BlendType.LUMINOSITY).convert("RGB")
    cases.append(dict(background=background.ravel().tolist(), foreground=foreground.ravel().tolist(),
                      output=np.asarray(output).ravel().tolist()))
print(json.dumps(cases))
