"""Generate trained Diffusers TAESD encoder references for even and odd RGB images."""
import pathlib
import sys
import numpy as np
import torch
from diffusers import AutoencoderTiny

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name in ('taesd', 'taesdxl'):
    model = AutoencoderTiny.from_pretrained(f'.superpowers/models/{name}', local_files_only=True).eval()
    for height, width in ((16, 24), (17, 19)):
        image = torch.tensor([np.sin(i * 0.037) for i in range(3 * height * width)], dtype=torch.float32).reshape(1, 3, height, width)
        with torch.no_grad():
            latent = model.encode(image).latents.contiguous().numpy().astype('<f4')
            reconstructed = model.decode(torch.from_numpy(latent)).sample.contiguous().numpy().astype('<f4')
        (out / f'{name}-encode-{height}x{width}.f32').write_bytes(latent.tobytes())
        (out / f'{name}-roundtrip-{height}x{width}.f32').write_bytes(reconstructed.tobytes())
        print(name, height, width, latent.shape, float(latent.min()), float(latent.max()))
