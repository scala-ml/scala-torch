"""Torch reference for the M-LSD large annotator (network defined here from the checkpoint keys; decoder from the documented algorithm).

usage: python reference_annotator_mlsd.py <annotator dir> <image> <out dir> [dilation=5]
Writes mlsd_ref.safetensors (input = 512x512 RGB CHW 0..255, output = tp_map 9x256x256 in float32, output_f64 = the same computed in float64 and rounded) and mlsd_segments.json
(segments for thr_v=0.1, thr_d=0.1, in 512 pixels) plus mlsd_lines_cv2.png (cv2.line thickness 1 render on 512x512).
"""
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from PIL import Image
from safetensors.torch import save_file

ann, image, out = sys.argv[1:4]
dil = int(sys.argv[4]) if len(sys.argv) > 4 else 5

def cbr(ci, co, k, p=0, d=1, act=nn.ReLU):
    return nn.Sequential(nn.Conv2d(ci, co, k, padding=p, dilation=d), nn.BatchNorm2d(co), act(inplace=True))

class Inv(nn.Module):
    def __init__(s, ci, co, st, t=6):
        super().__init__(); h = ci * t; s.res = st == 1 and ci == co
        s.conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(ci, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU6()),
            nn.Sequential(nn.Conv2d(h, h, 3, st, 1, groups=h, bias=False), nn.BatchNorm2d(h), nn.ReLU6()),
            nn.Conv2d(h, co, 1, bias=False), nn.BatchNorm2d(co))
    def forward(s, x): return x + s.conv(x) if s.res else s.conv(x)

class Inv1(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU6()),
                               nn.Conv2d(32, 16, 1, bias=False), nn.BatchNorm2d(16))
    def forward(s, x): return s.conv(x)

class A(nn.Module):
    def __init__(s, c1, c2, up):
        super().__init__(); s.conv1 = cbr(c2, 64, 1); s.conv2 = cbr(c1, 64, 1); s.up = up
    def forward(s, a, b):
        b = s.conv1(b); a = s.conv2(a)
        if s.up: b = F.interpolate(b, scale_factor=2.0, mode="bilinear", align_corners=True)
        return torch.cat((a, b), 1)

class B(nn.Module):
    def __init__(s):
        super().__init__(); s.conv1 = cbr(128, 128, 3, 1); s.conv2 = cbr(128, 64, 3, 1)
    def forward(s, x): return s.conv2(s.conv1(x) + x)

class C(nn.Module):
    def __init__(s):
        super().__init__(); s.conv1 = cbr(64, 64, 3, dil, dil); s.conv2 = cbr(64, 64, 3, 1); s.conv3 = nn.Conv2d(64, 16, 1)
    def forward(s, x): return s.conv3(s.conv2(s.conv1(x)))

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        cfg = [(16,24,2),(24,24,1),(24,32,2),(32,32,1),(32,32,1),(32,64,2),(64,64,1),(64,64,1),(64,64,1),(64,96,1),(96,96,1),(96,96,1)]
        feats = [nn.Sequential(nn.Conv2d(4, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6()), Inv1()] + [Inv(*c) for c in cfg]
        s.backbone = nn.Module(); s.backbone.features = nn.ModuleList(feats)
        s.block15 = A(64, 96, False); s.block16 = B(); s.block17 = A(32, 64, True); s.block18 = B()
        s.block19 = A(24, 64, True); s.block20 = B(); s.block21 = A(16, 64, True); s.block22 = B(); s.block23 = C()
    def forward(s, x):
        taps = {}
        for i, f in enumerate(s.backbone.features):
            x = f(x)
            if i in (1, 3, 6, 10, 13): taps[i] = x
        x = s.block16(s.block15(taps[10], taps[13]))
        x = s.block18(s.block17(taps[6], x))
        x = s.block20(s.block19(taps[3], x))
        x = s.block22(s.block21(taps[1], x))
        return s.block23(x)[:, 7:]

net = Net(); net.load_state_dict(torch.load(f"{ann}/mlsd_large_512_fp32.pth", map_location="cpu"), strict=True); net.eval()
img = np.asarray(Image.open(image).convert("RGB"))
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
chw = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous()
inp = torch.cat([chw, torch.ones(1, 512, 512)], 0)
with torch.no_grad():
    tp = net(inp[None] / 127.5 - 1.0)
net64 = net.double()
with torch.no_grad():
    tp64 = net64(inp[None].double() / 127.5 - 1.0)
rel32 = float((tp[0].double() - tp64[0]).abs().max() / tp64[0].abs().max())
print("torch float32 vs float64 tp_map relative deviation", rel32)
save_file({"input": chw, "output": tp[0].contiguous(), "output_f64": tp64[0].float().contiguous()}, f"{out}/mlsd_ref.safetensors")

# documented decoder
t = tp[0]; heat = torch.sigmoid(t[0])
hmax = F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
heat = heat * (hmax == heat).float()
scores, idx = torch.topk(heat.reshape(-1), 200)
vmap = t[1:5].permute(1, 2, 0)
segs = []
for sc, i in zip(scores.tolist(), idx.tolist()):
    y, x = divmod(i, 256); v = vmap[y, x]
    d = float(torch.sqrt(((v[:2] - v[2:]) ** 2).sum()))
    if sc > 0.1 and d > 0.1:
        segs.append([sc] + [2 * float(c) for c in (x + v[0], y + v[1], x + v[2], y + v[3])])
json.dump(segs, open(f"{out}/mlsd_segments.json", "w"))
canvas = np.zeros((512, 512, 3), np.uint8)
for _, a, b, c, d in segs: cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 1)
Image.fromarray(canvas).save(f"{out}/mlsd_lines_cv2.png")
print("segments", len(segs), "tp range", float(tp.min()), float(tp.max()))
