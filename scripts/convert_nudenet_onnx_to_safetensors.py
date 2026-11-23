"""One-off offline converter: NudeNet v3 320n.onnx -> plain float32 safetensors (+ .json sidecar with the class names).

usage: python convert_nudenet_onnx_to_safetensors.py <320n.onnx> [<out.safetensors>]
The ONNX file is an Ultralytics YOLOv8n export (opset with Conv+BatchNorm already fused): every `Conv` initializer already carries the
YOLOv8 module name (`model.N....conv.weight` / `.conv.bias`, the 1x1 head convs `model.22.cv{2,3}.i.2.weight|bias`, the DFL ramp
`model.22.dfl.conv.weight`). Those initializers are copied verbatim; shape-only helper constants are dropped. The Scala loader
(scalatorch.models.YoloV8) detects the fused layout (a `conv.bias` and no `bn.*`) and validates every name and shape strictly.
Only needs `onnx`, `numpy`, `safetensors`.

Licence: NudeNet (notAI-tech/NudeNet) weights are AGPL-3.0 (the ONNX metadata says "AGPL-3.0 License"); the converted file inherits that.
"""
import sys, json, ast
import numpy as np, onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".safetensors"
m = onnx.load(src)
meta = {p.key: p.value for p in m.metadata_props}
names = ast.literal_eval(meta["names"])
out = {}
for init in m.graph.initializer:
    if init.name.startswith("model."):
        out[init.name] = np.ascontiguousarray(numpy_helper.to_array(init).astype(np.float32))
# every Conv node must be fed by two initializers we kept (weight, bias)
for n in m.graph.node:
    if n.op_type == "Conv":
        assert all(i in out for i in n.input[1:]), f"conv {n.name} uses an initializer that is not a module weight"
save_file(out, dst, metadata={"format": "onnx-fused", "source": src.replace("\\", "/").split("/")[-1], "task": meta.get("task", "detect")})
with open(dst + ".json", "w", encoding="utf-8") as f:
    json.dump({"names": {str(k): v for k, v in names.items()}, "imgsz": meta.get("imgsz"), "license": meta.get("license")}, f, indent=1)
print(dst, len(out), "tensors,", len(names), "classes")
