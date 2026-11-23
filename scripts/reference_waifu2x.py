"""torch reference for the Scala waifu2x upconv_7 path (independent oracle, written from the documented layer list; no waifu2x code is used).

usage: python reference_waifu2x.py <waifu2x dir with {art,photo}-noiseN.json> <image> <out dir> [crop=128]
Reads the original Torch7 JSON model (not the converted safetensors), takes a crop x crop RGB crop from the image centre, replicate-pads
by 7, runs 6x [conv3x3 valid + LeakyReLU(0.1)] then ConvTranspose2d(256,3,4,stride 2,padding 3), no clamping, float32.
Writes <out>/waifu2x_ref.safetensors {input (3,c,c), <style>_n<noise> (3,2c,2c)} for art noise 1 and photo noise 3 (and art 0/2/3, photo 0..2).
"""
import json, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from safetensors.numpy import save_file

src, image, out = sys.argv[1:4]; crop = int(sys.argv[4]) if len(sys.argv) > 4 else 128
im = Image.open(image).convert("RGB"); W, H = im.size
l, t = (W - crop) // 2, (H - crop) // 2
x = torch.from_numpy(np.asarray(im.crop((l, t, l + crop, t + crop)), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].contiguous()
res = {"input": x[0].numpy().copy()}
torch.set_grad_enabled(False)
for style in ("art", "photo"):
    for noise in range(4):
        layers = json.load(open(f"{src}/{style}-noise{noise}.json", encoding="utf-8"))
        v = F.pad(x, (7, 7, 7, 7), mode="replicate")
        for lay in layers[:6]:
            v = F.leaky_relu(F.conv2d(v, torch.tensor(lay["weight"], dtype=torch.float32), torch.tensor(lay["bias"], dtype=torch.float32)), 0.1)
        d = layers[6]
        v = F.conv_transpose2d(v, torch.tensor(d["weight"], dtype=torch.float32), None, stride=2, padding=3)
        assert v.shape == (1, 3, 2 * crop, 2 * crop), v.shape
        res[f"{style}_n{noise}"] = np.ascontiguousarray(v[0].numpy())
save_file(res, f"{out}/waifu2x_ref.safetensors")
print("ok", {k: v.shape for k, v in res.items()})
