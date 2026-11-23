"""One-off offline converter: rembg's u2net.onnx (U2-Net full, Qin et al. 2020) -> plain float32 safetensors with module-style names.

usage: python convert_u2net_onnx_to_safetensors.py <u2net.onnx> [<out.safetensors>]
The ONNX export has Conv+BatchNorm already fused and generic initializer names (onnx::Conv_NNN / numbers) for the RSU convs; only
`side*.weight|bias` and `outconv.weight|bias` keep their names. The 119 `Conv` nodes are visited in graph (= forward) order and matched,
one by one, against the documented U2-Net layout below (stage1..6 encoders, stage5d..1d decoders, side1..6, outconv); input channels,
output channels, kernel size and dilation of every node are asserted, so a different graph fails loudly instead of being mis-named.
Output names: `<stage>.<layer>.weight|bias`, e.g. `stage1.rebnconvin.weight`, `stage5d.rebnconv3d.bias`, `side3.weight`, `outconv.bias`.
Only needs `onnx`, `numpy`, `safetensors`.

Licence: the U2-Net code (xuebinqin/U-2-Net) is Apache-2.0; the weights were published by the U2-Net authors and re-hosted by rembg
(danielgatis/rembg, MIT code). No explicit weight licence statement was found in the ONNX metadata; see the notices in docs.
"""
import sys
import numpy as np, onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

def rsu(name, n, cin, mid, cout):
    """RSU-n (n = 7,6,5,4) encoder-decoder block: [(layer, ci, co, dilation)] in forward order."""
    L = [("rebnconvin", cin, cout, 1), ("rebnconv1", cout, mid, 1)]
    for i in range(2, n): L.append((f"rebnconv{i}", mid, mid, 1))        # pooled levels 2..n-1
    L.append((f"rebnconv{n}", mid, mid, 2))                                # dilated bottom
    for i in range(n - 1, 1, -1): L.append((f"rebnconv{i}d", 2 * mid, mid, 1))
    L.append(("rebnconv1d", 2 * mid, cout, 1))
    return [(f"{name}.{l}", ci, co, d, 3) for l, ci, co, d in L]

def rsu4f(name, cin, mid, cout):
    L = [("rebnconvin", cin, cout, 1), ("rebnconv1", cout, mid, 1), ("rebnconv2", mid, mid, 2), ("rebnconv3", mid, mid, 4), ("rebnconv4", mid, mid, 8),
         ("rebnconv3d", 2 * mid, mid, 4), ("rebnconv2d", 2 * mid, mid, 2), ("rebnconv1d", 2 * mid, cout, 1)]
    return [(f"{name}.{l}", ci, co, d, 3) for l, ci, co, d in L]

LAYOUT = (rsu("stage1", 7, 3, 32, 64) + rsu("stage2", 6, 64, 32, 128) + rsu("stage3", 5, 128, 64, 256) + rsu("stage4", 4, 256, 128, 512)
          + rsu4f("stage5", 512, 256, 512) + rsu4f("stage6", 512, 256, 512)
          + rsu4f("stage5d", 1024, 256, 512) + rsu("stage4d", 4, 1024, 128, 256) + rsu("stage3d", 5, 512, 64, 128)
          + rsu("stage2d", 6, 256, 32, 64) + rsu("stage1d", 7, 128, 16, 64)
          + [(f"side{i + 1}", c, 1, 1, 3) for i, c in enumerate([64, 64, 128, 256, 512, 512])] + [("outconv", 6, 1, 1, 1)])

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".safetensors"
m = onnx.load(src)
init = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in m.graph.initializer}
convs = [n for n in m.graph.node if n.op_type == "Conv"]
assert len(convs) == len(LAYOUT) == 119, (len(convs), len(LAYOUT))
out = {}
for node, (name, ci, co, dil, k) in zip(convs, LAYOUT):
    w, b = init[node.input[1]], init[node.input[2]]
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    assert w.shape == (co, ci, k, k) and b.shape == (co,), (name, w.shape, b.shape, (co, ci, k, k))
    assert list(attrs["dilations"]) == [dil, dil] and list(attrs["pads"]) == [dil * (k // 2)] * 4 and attrs["group"] == 1, (name, attrs)
    if node.input[1].startswith(("side", "outconv")): assert node.input[1] == name + ".weight", (node.input[1], name)
    out[name + ".weight"] = np.ascontiguousarray(w); out[name + ".bias"] = np.ascontiguousarray(b)
save_file(out, dst, metadata={"format": "onnx-fused", "source": src.replace("\\", "/").split("/")[-1], "model": "u2net"})
print(dst, len(out), "tensors")
