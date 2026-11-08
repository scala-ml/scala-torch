"""Torch reference for the Scala YoloV8 detector (uses the `ultralytics` package only as an independent oracle; AGPL-3.0, not copied).

usage: python reference_yolov8.py <model.pt> <image> <out dir> [conf=0.3]
Writes <out>/<stem>_ref.safetensors:
  input            letterboxed network input, CHW float32 in 0..1 (the exact tensor fed to both implementations)
  box0..2 cls0..2 coef0..2 proto   raw head outputs (CHW, float32 run);   *_f64 variants from a float64 run
  pred             Ultralytics' decoded output (4 xywh + nc + nm, N) of the float32 run
and <out>/<stem>_ref.json with predict() results: boxes (xyxy, source pixels), conf, cls, letterbox geometry,
plus <stem>_masks.npy (n, H, W) uint8 masks scaled to the source image when the model is a segmenter.
"""
import sys, json, numpy as np, torch, cv2
from safetensors.torch import save_file
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

pt, image, out = sys.argv[1:4]
conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
stem = pt.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]

yolo = YOLO(pt)
net = yolo.model.float().eval()
head = net.model[-1]
bgr = cv2.imread(image)
lb = LetterBox(new_shape=(640, 640), auto=True, stride=32)
padded = lb(image=bgr)
x = torch.from_numpy(np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))).float()[None] / 255.0

def run(model, x):
    cap = {}
    hooks = []
    for i in range(3):
        hooks.append(model.model[-1].cv2[i][2].register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f"box{i}", o.detach())))
        hooks.append(model.model[-1].cv3[i][2].register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f"cls{i}", o.detach())))
        if hasattr(model.model[-1], "cv4"):
            hooks.append(model.model[-1].cv4[i][2].register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f"coef{i}", o.detach())))
    if hasattr(model.model[-1], "proto"):
        hooks.append(model.model[-1].proto.register_forward_hook(lambda m, a, o: cap.__setitem__("proto", o.detach())))
    with torch.no_grad():
        y = model(x)
    for h in hooks: h.remove()
    y = y[0] if isinstance(y, (tuple, list)) else y
    y = y[0] if isinstance(y, (tuple, list)) else y
    return cap, y

cap32, y32 = run(net, x)
import copy
net64 = copy.deepcopy(net).double()
cap64, y64 = run(net64, x.double())
tensors = {"input": x[0].contiguous(), "pred": y32[0].contiguous()}
for k, v in cap32.items(): tensors[k] = v[0].contiguous()
for k, v in cap64.items(): tensors[k + "_f64"] = v[0].float().contiguous()
tensors["pred_f64"] = y64[0].float().contiguous()
save_file(tensors, f"{out}/{stem}_ref.safetensors")

res = yolo.predict(image, conf=conf, iou=0.7, imgsz=640, verbose=False)[0]
h, w = bgr.shape[:2]
info = {"conf": conf, "orig": [w, h], "input_hw": list(x.shape[2:]),
        "boxes": res.boxes.xyxy.tolist(), "scores": res.boxes.conf.tolist(), "cls": res.boxes.cls.tolist()}
if res.masks is not None:
    m = res.masks.data.cpu().numpy()  # letterboxed resolution
    ih, iw = m.shape[1:]
    gain = min(640 / h, 640 / w)
    nw, nh = round(w * gain), round(h * gain)
    dw, dh = (640 - nw) % 32 / 2, (640 - nh) % 32 / 2
    top, left = round(dh - 0.1), round(dw - 0.1)
    full = np.stack([(cv2.resize(mm[top:top + nh, left:left + nw].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8) * 255 for mm in m]) if len(m) else np.zeros((0, h, w), np.uint8)
    np.save(f"{out}/{stem}_masks.npy", full)
with open(f"{out}/{stem}_ref.json", "w") as f: json.dump(info, f)
print(stem, "boxes", len(info["boxes"]), "pred", tuple(y32.shape), "max|f32-f64| pred", float((y32.double() - y64).abs().max()))
