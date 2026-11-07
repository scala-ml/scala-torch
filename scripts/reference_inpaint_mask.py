"""Pillow-generated reference masks; never used by the native runtime."""
import json
import sys
from PIL import Image, ImageFilter

width, height = 13, 9
patterns = {
    "edge": [255 if x >= 6 else 0 for y in range(height) for x in range(width)],
    "gradient": [(x * 37 + y * 53 + x * y * 7) % 256 for y in range(height) for x in range(width)],
    "point": [255 if (x, y) == (0, 4) else 0 for y in range(height) for x in range(width)],
}
cases = []
for name, values in patterns.items():
    image = Image.new("L", (width, height))
    image.putdata(values)
    for radius in (0, 1, 2, 4, 7, 64):
        blurred = image.filter(ImageFilter.GaussianBlur(radius)) if radius else image
        cases.append(dict(pattern=name, radius=radius, input=values, output=list(blurred.getdata())))
json.dump(dict(width=width, height=height, cases=cases), sys.stdout, separators=(",", ":"))
