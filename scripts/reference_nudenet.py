"""onnxruntime reference for the Scala NudeNet path (independent oracle; NudeNet / Ultralytics are AGPL-3.0, nothing copied).

usage: python reference_nudenet.py <320n.onnx> <image> <out dir>
Preprocessing is the documented NudeNet one: pad the image to a square with zeros at the bottom/right, resize to 320x320 (bilinear), RGB, 0..1.
Writes <out>/nudenet_ref.safetensors {input (3,320,320), output (22,2100)} and <out>/nudenet_ref.json {orig, boxes (xyxy source px), scores, cls}
after the documented post-processing (best class score >= 0.2, class-agnostic NMS IoU 0.45).
"""
import sys, json, numpy as np, cv2, onnxruntime as ort
from safetensors.numpy import save_file

onnx_path, image, out = sys.argv[1:4]
bgr = cv2.imread(image); h, w = bgr.shape[:2]; s = max(h, w)
padded = cv2.copyMakeBorder(bgr, 0, s - h, 0, s - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
small = cv2.resize(padded, (320, 320), interpolation=cv2.INTER_LINEAR)
x = (small[..., ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
y = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]  # (22, 2100)
save_file({"input": x[0].copy(), "output": np.ascontiguousarray(y)}, f"{out}/nudenet_ref.safetensors")
det = y.T; sc = det[:, 4:].max(1); cl = det[:, 4:].argmax(1); keep = sc >= 0.2
b = det[keep, :4]; sc = sc[keep]; cl = cl[keep]; f = s / 320.0
xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * f, (b[:, 1] - b[:, 3] / 2) * f, (b[:, 0] + b[:, 2] / 2) * f, (b[:, 1] + b[:, 3] / 2) * f], 1)
idx = cv2.dnn.NMSBoxes([[float(a), float(bb), float(c - a), float(d - bb)] for a, bb, c, d in xyxy], sc.tolist(), 0.2, 0.45)
idx = [int(i) for i in np.array(idx).reshape(-1)]
res = {"orig": [w, h], "boxes": [[float(v) for v in np.clip(xyxy[i], 0, [w, h, w, h])] for i in idx], "scores": [float(sc[i]) for i in idx], "cls": [int(cl[i]) for i in idx]}
json.dump(res, open(f"{out}/nudenet_ref.json", "w"))
print("detections", len(idx), [(int(cl[i]), round(float(sc[i]), 3)) for i in idx])
