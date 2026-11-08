"""Torch reference for the HED / lineart / lineart_anime annotator networks (networks defined here from the checkpoint keys).

usage: python reference_annotator_lineart.py <annotator dir> <image> <out dir>
Writes hed_ref.safetensors, lineart_ref.safetensors, anime_ref.safetensors (input + output, float32, before 8-bit quantisation).
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

ann, image, out = sys.argv[1:4]
img = np.asarray(Image.open(image).convert("RGB")).astype(np.float32)  # H,W,3 RGB 0..255
H, W, _ = img.shape
chw = torch.from_numpy(img).permute(2, 0, 1).contiguous()

# ---------------------------------------------------------------- HED
sd = {k: v.float() for k, v in torch.load(f"{ann}/ControlNetHED.pth", map_location="cpu").items()}
def hed(x):
    x = x - sd["norm"]
    sides = []
    for b, n in zip(range(1, 6), (2, 2, 3, 3, 3)):
        if b > 1: x = F.max_pool2d(x, 2, 2)
        for i in range(n):
            x = F.relu(F.conv2d(x, sd[f"block{b}.convs.{i}.weight"], sd[f"block{b}.convs.{i}.bias"], padding=1))
        sides.append(F.conv2d(x, sd[f"block{b}.projection.weight"], sd[f"block{b}.projection.bias"]))
    ups = [F.interpolate(s, size=(H, W), mode="bilinear", align_corners=False) for s in sides]
    return torch.sigmoid(torch.stack(ups).mean(0))[0, 0]
with torch.no_grad():
    save_file({"input": chw, "output": hed(chw[None]).contiguous()}, f"{out}/hed_ref.safetensors")

# ---------------------------------------------------------------- lineart (ResNet generator)
sd = {k: v.float() for k, v in torch.load(f"{ann}/sk_model.pth", map_location="cpu").items()}
inorm = lambda x: F.instance_norm(x, eps=1e-5)
def conv(x, n, **kw): return F.conv2d(x, sd[n + ".weight"], sd[n + ".bias"], **kw)
def lineart(x):
    x = F.relu(inorm(conv(F.pad(x, (3,) * 4, mode="reflect"), "model0.1")))
    x = F.relu(inorm(conv(x, "model1.0", stride=2, padding=1)))
    x = F.relu(inorm(conv(x, "model1.3", stride=2, padding=1)))
    for i in range(3):
        y = F.relu(inorm(conv(F.pad(x, (1,) * 4, mode="reflect"), f"model2.{i}.conv_block.1")))
        y = inorm(conv(F.pad(y, (1,) * 4, mode="reflect"), f"model2.{i}.conv_block.5"))
        x = x + y
    for n in ("model3.0", "model3.3"):
        x = F.relu(inorm(F.conv_transpose2d(x, sd[n + ".weight"], sd[n + ".bias"], stride=2, padding=1, output_padding=1)))
    return torch.sigmoid(conv(F.pad(x, (3,) * 4, mode="reflect"), "model4.1"))[0, 0]
with torch.no_grad():
    save_file({"input": (chw / 255.0).contiguous(), "output": lineart((chw / 255.0)[None]).contiguous()}, f"{out}/lineart_ref.safetensors")

# ---------------------------------------------------------------- lineart_anime (pix2pix U-Net)
sd = {k: v.float() for k, v in torch.load(f"{ann}/netG.pth", map_location="cpu").items()}
def block(x, p, depth):
    inner = depth == 6
    y = F.conv2d(F.leaky_relu(x, 0.2), sd[p + ".1.weight"], sd[p + ".1.bias"], stride=2, padding=1)
    if inner:
        y = F.conv_transpose2d(F.relu(y), sd[p + ".3.weight"], sd[p + ".3.bias"], stride=2, padding=1)
    else:
        y = inorm(y)
        y = block(y, p + ".3.model", depth + 1)
        y = F.conv_transpose2d(F.relu(y), sd[p + ".5.weight"], sd[p + ".5.bias"], stride=2, padding=1)
    return torch.cat([x, inorm(y)], 1)
def anime(x):
    r = "module.model.model"
    h = F.conv2d(x, sd[r + ".0.weight"], sd[r + ".0.bias"], stride=2, padding=1)
    h = block(h, r + ".1.model", 0)
    return torch.tanh(F.conv_transpose2d(F.relu(h), sd[r + ".3.weight"], sd[r + ".3.bias"], stride=2, padding=1))[0, 0]
with torch.no_grad():
    xin = (chw / 127.5 - 1.0)
    save_file({"input": xin.contiguous(), "output": anime(xin[None]).contiguous()}, f"{out}/anime_ref.safetensors")
print("ok")
