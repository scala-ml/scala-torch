"""Generate fixed-latent TAESD decoder references from local Diffusers models."""
import pathlib
import sys
import numpy as np
import torch
from diffusers import AutoencoderTiny

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
latent = torch.tensor([np.sin(i * 0.37) * 2 for i in range(24)], dtype=torch.float32).reshape(1, 4, 2, 3)
for name in ('taesd', 'taesdxl'):
    model = AutoencoderTiny.from_pretrained(f'.superpowers/models/{name}', local_files_only=True).eval()
    with torch.no_grad():
        decoded = model.decode(latent).sample.contiguous().numpy().astype('<f4')
    (out / f'{name}-decode.f32').write_bytes(decoded.tobytes())
    print(name, decoded.shape, float(decoded.min()), float(decoded.max()))
