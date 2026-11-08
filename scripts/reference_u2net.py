"""onnxruntime reference for the Scala U2-Net path (independent oracle; U2-Net is Apache-2.0, nothing copied).

usage: python reference_u2net.py <u2net.onnx> <image> <out dir>
Preprocessing (documented rembg/U2-Net one): RGB, resize to 320x320 (INTER_AREA here), divide by the image maximum, ImageNet mean/std.
Writes <out>/u2net_ref.safetensors {input (3,320,320), d0..d6 (320,320) sigmoid side maps, matte (320,320) = (d0-min)/(max-min)}.
"""
import sys, numpy as np, cv2, onnxruntime as ort
from safetensors.numpy import save_file

onnx_path, image, out = sys.argv[1:4]
rgb = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
small = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
small = small / max(float(small.max()), 1e-6)
x = ((small - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)[None].astype(np.float32)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ys = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x)})
t = {"input": np.ascontiguousarray(x[0])}
for i, y in enumerate(ys): t[f"d{i}"] = np.ascontiguousarray(y[0, 0])
d0 = ys[0][0, 0]; t["matte"] = np.ascontiguousarray((d0 - d0.min()) / (d0.max() - d0.min()))
save_file(t, f"{out}/u2net_ref.safetensors")
print("d0 range", float(d0.min()), float(d0.max()))
