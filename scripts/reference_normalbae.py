"""Torch reference for the NormalBAE surface-normal network used by ControlNet's `normal_bae` annotator
(scannet.pt from lllyasviel/Annotators). The network is defined here from the checkpoint keys and the documented architecture:
a tf_efficientnet_b5 encoder (TF 'same' padding, BatchNorm eps 1e-3, SiLU, squeeze-excite with in_chs/4 channels) whose block
outputs (and the raw 1x1 conv_head output) feed a 4-stage bilinear-upsample + conv/BN/LeakyReLU decoder, followed by per-pixel
MLP heads at 1/4, 1/2 and 1/1 resolution (test mode: all pixels refined).

usage: python reference_normalbae.py <scannet.pt> <image> <out.safetensors> [<size> | <width>x<height>]
The image is resized (area) to <size> x <size> or <width> x <height> (default 256; multiples of 32), normalised with the ImageNet mean/std and
run through the network. Writes `input` (3,H,W) and `output` (4,H,W: normal xyz + kappa, before any 8-bit quantisation).
Only needs torch, numpy, pillow, safetensors.
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

ckpt, image, out = sys.argv[1:4]
spec = sys.argv[4] if len(sys.argv) > 4 else "256"
W, H = (int(v) for v in (spec.split("x") if "x" in spec else (spec, spec)))
sd = torch.load(ckpt, map_location="cpu")["model"]
sd = {k[7:] if k.startswith("module.") else k: v.float() for k, v in sd.items()}
P = "encoder.original_model."

def same_pad(x, k, s):
    h, w = x.shape[-2:]
    ph = max((-(-h // s) - 1) * s + k - h, 0); pw = max((-(-w // s) - 1) * s + k - w, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))

def conv_same(x, w, s=1, groups=1):
    return F.conv2d(same_pad(x, w.shape[-1], s), w, stride=s, groups=groups)

def bn(x, n, eps):
    return F.batch_norm(x, sd[n + ".running_mean"], sd[n + ".running_var"], sd[n + ".weight"], sd[n + ".bias"], False, 0.0, eps)

def se(x, n):
    y = x.mean((2, 3), keepdim=True)
    y = F.silu(F.conv2d(y, sd[n + ".conv_reduce.weight"], sd[n + ".conv_reduce.bias"]))
    y = F.conv2d(y, sd[n + ".conv_expand.weight"], sd[n + ".conv_expand.bias"])
    return x * torch.sigmoid(y)

# (repeats, kernel, stride, out channels, ds-block); block 0 is depthwise-separable, the rest inverted residual (expand x6)
STAGES = [(3, 3, 1, 24, True), (5, 3, 2, 40, False), (5, 5, 2, 64, False), (7, 3, 2, 128, False), (7, 5, 1, 176, False), (9, 5, 2, 304, False), (3, 3, 1, 512, False)]
EPS = 1e-3
def encoder(x):
    feats = [x]
    x = F.silu(bn(conv_same(x, sd[P + "conv_stem.weight"], 2), P + "bn1", EPS))
    feats += [x, x, x]
    cin = 48
    for si, (n, k, s0, cout, ds) in enumerate(STAGES):
        for bi in range(n):
            p = f"{P}blocks.{si}.{bi}"; s = s0 if bi == 0 else 1
            inp = x
            if ds:
                x = F.silu(bn(conv_same(x, sd[p + ".conv_dw.weight"], s, cin), p + ".bn1", EPS))
                x = se(x, p + ".se")
                x = bn(conv_same(x, sd[p + ".conv_pw.weight"]), p + ".bn2", EPS)
            else:
                x = F.silu(bn(conv_same(x, sd[p + ".conv_pw.weight"]), p + ".bn1", EPS))
                x = F.silu(bn(conv_same(x, sd[p + ".conv_dw.weight"], s, x.shape[1]), p + ".bn2", EPS))
                x = se(x, p + ".se")
                x = bn(conv_same(x, sd[p + ".conv_pwl.weight"]), p + ".bn3", EPS)
            if s == 1 and cin == cout: x = x + inp
            cin = cout
        feats.append(x)
    feats.append(conv_same(x, sd[P + "conv_head.weight"]))  # features[11]: raw conv_head output (no BN / activation)
    return feats

def norm_normalize(o):
    nx, ny, nz, k = o.split(1, 1)
    n = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-10
    return torch.cat([nx / n, ny / n, nz / n, F.elu(k) + 1.0 + 0.01], 1)

def up(x, skip, n):
    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
    x = torch.cat([x, skip], 1)
    for c, b in ((0, 1), (3, 4)):
        x = F.leaky_relu(bn(F.conv2d(x, sd[f"decoder.{n}._net.{c}.weight"], sd[f"decoder.{n}._net.{c}.bias"], padding=1), f"decoder.{n}._net.{b}", 1e-5))
    return x

def head(x, n):
    for i in (0, 2, 4):
        x = F.relu(F.conv1d(x, sd[f"decoder.{n}.{i}.weight"], sd[f"decoder.{n}.{i}.bias"]))
    return F.conv1d(x, sd[f"decoder.{n}.6.weight"], sd[f"decoder.{n}.6.bias"])

def refine(feat, prev, n):
    f = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=True)
    p = F.interpolate(prev, scale_factor=2, mode="bilinear", align_corners=True)
    B, _, H, W = f.shape
    return norm_normalize(head(torch.cat([f, p], 1).view(B, -1, H * W), n)).view(B, 4, H, W)

def network(x):
    f = encoder(x)
    d0 = F.conv2d(f[11], sd["decoder.conv2.weight"], sd["decoder.conv2.bias"])
    d1 = up(d0, f[8], "up1"); d2 = up(d1, f[6], "up2"); d3 = up(d2, f[5], "up3"); d4 = up(d3, f[4], "up4")
    r8 = norm_normalize(F.conv2d(d2, sd["decoder.out_conv_res8.weight"], sd["decoder.out_conv_res8.bias"], padding=1))
    r4 = refine(d2, r8, "out_conv_res4"); r2 = refine(d3, r4, "out_conv_res2"); r1 = refine(d4, r2, "out_conv_res1")
    return r1

img = Image.open(image).convert("RGB").resize((W, H), Image.BOX)
x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
x = ((x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]).contiguous()
with torch.no_grad():
    o = network(x[None])[0].contiguous()
print("output", tuple(o.shape), "xyz range", o[:3].min().item(), o[:3].max().item(), "kappa", o[3].min().item(), o[3].max().item())
save_file({"input": x, "output": o}, out)
