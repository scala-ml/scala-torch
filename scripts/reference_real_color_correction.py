"""Generate color-correction reference PNGs for two user-provided images."""
import sys
from pathlib import Path
from PIL import Image

nui, source, reference, output = map(Path, sys.argv[1:5])
sys.path.insert(0, str(nui / "python"))
from sdfx.color_correction import a1111, adain, wavelet

output.mkdir(parents=True, exist_ok=True)
with Image.open(source) as src, Image.open(reference) as ref:
    for name, fn in (("A1111", a1111), ("AdaIN", adain), ("Wavelet", wavelet)):
        fn(src, ref, .4).save(output / f"{name}.png")
