"""Generate NUI VAE Tools' RGBA/CMYK latent packing and unpacking fixtures."""
import pathlib
import sys
import numpy as np
sys.path.insert(0, 'C:/Users/Mizushima/dev/stablediffusion-nui/python')
from sdfx.vae_tools import pack, unpack

output = pathlib.Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
for channels in (4, 16):
    values = np.linspace(-12, 12, channels * 2 * 3, dtype=np.float32).reshape(1, channels, 2, 3)
    for representation in ('RGBA', 'CMYK'):
        image, stats = pack(values, representation)
        restored = unpack(image, representation, channels).astype('<f4')
        stem = f'{representation.lower()}-{channels}'
        (output / f'{stem}.bytes').write_bytes(image.tobytes())
        (output / f'{stem}.f32').write_bytes(restored.tobytes())
        if representation == 'RGBA':
            image.save(output / f'{stem}.png')
        print(stem, image.size, stats['clipped_values'])
