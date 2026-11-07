"""Generate a NUI color-enhance reference PNG and metadata."""
import json
import sys
from pathlib import Path
from PIL import Image

nui, source, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(nui / "python"))
from sdfx.image_tools import color_enhance

output.mkdir(parents=True, exist_ok=True)
with Image.open(source) as image:
    result, info = color_enhance(image, 1.0, 99.5)
    result.save(output / "result.png")
    (output / "info.json").write_text(json.dumps(info), encoding="utf-8")
