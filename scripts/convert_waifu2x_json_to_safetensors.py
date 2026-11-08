"""One-off offline converter: nagadomi/waifu2x (Torch7) UpConv7 `*_scale2.0x_model.json` -> float32 safetensors.

usage: python convert_waifu2x_json_to_safetensors.py <in dir with {art,photo}-noise{0..3}.json> [<out dir>]
Each JSON is a list of exactly 7 layers: six `nn.SpatialConvolutionMM` 3x3 (3-16-32-64-128-128-256, no padding) and a final
`nn.SpatialFullConvolution` 256->3, 4x4, stride 2, pad 3 (weight layout in x out x kH x kW, bias all zero). Shapes, layer classes and
strides are asserted, so a different architecture fails loudly. Output names: `conv{0..5}.weight|bias`, `deconv.weight|bias`.
Only needs numpy and safetensors.

Licence: the waifu2x code (nagadomi/waifu2x) is MIT; its model files are published inside that repository without a separate
licence statement (the repository licence is MIT, models were trained on the author's own data set; treat as MIT-by-repository,
unverified). Pinned upstream commit cc385f97a9debfe611316aabfd5d8bb30ba2dbeb, models/upconv_7/{art,photo}/noise{0..3}_scale2.0x_model.json
(the same files python/sdfx/waifu2x_assets.json in the NUI repository pins by SHA-256).
"""
import json, sys, pathlib
import numpy as np
from safetensors.numpy import save_file

src = pathlib.Path(sys.argv[1]); dst = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
CH = (3, 16, 32, 64, 128, 128, 256)
for style in ("art", "photo"):
    for noise in range(4):
        layers = json.loads((src / f"{style}-noise{noise}.json").read_text(encoding="utf-8"))
        assert len(layers) == 7, len(layers)
        out = {}
        for i, l in enumerate(layers):
            w = np.asarray(l["weight"], dtype=np.float32); b = np.asarray(l["bias"], dtype=np.float32)
            if i < 6:
                assert l["class_name"] == "nn.SpatialConvolutionMM" and (l["kW"], l["kH"], l["dW"], l["dH"], l["padW"], l["padH"]) == (3, 3, 1, 1, 0, 0), l["class_name"]
                assert w.shape == (CH[i + 1], CH[i], 3, 3) and b.shape == (CH[i + 1],), (i, w.shape)
                out[f"conv{i}.weight"], out[f"conv{i}.bias"] = w, b
            else:
                assert l["class_name"] == "nn.SpatialFullConvolution" and (l["kW"], l["dW"], l["padW"]) == (4, 2, 3)
                assert w.shape == (256, 3, 4, 4) and b.shape == (3,) and not b.any(), (w.shape, b.shape)
                out["deconv.weight"], out["deconv.bias"] = w, b
            assert np.isfinite(w).all() and np.isfinite(b).all()
        out = {k: np.ascontiguousarray(v) for k, v in out.items()}
        save_file(out, str(dst / f"{style}-noise{noise}.safetensors"), metadata={"model": "waifu2x-upconv_7", "style": style, "noise": str(noise), "scale": "2"})
        print(style, noise, "ok")
