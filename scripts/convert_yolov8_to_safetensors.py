"""One-off offline converter: Ultralytics YOLOv8 .pt checkpoint -> plain float32 safetensors state dict (+ a .json sidecar).

usage: python convert_yolov8_to_safetensors.py <file.pt> [<file.pt> ...]
Writes <file>.safetensors next to each input. The Scala runtime (scalatorch.models.YoloV8) reads only the safetensors; it does
not need Python or Ultralytics. Tensors keep their checkpoint names (Conv + BatchNorm stay separate; Scala folds BN at load).
Float16 checkpoint tensors are widened to float32 exactly. BatchNorm num_batches_tracked bookkeeping is dropped.

Loading the pickle needs the `ultralytics` package (it is a pickle of Ultralytics classes) or, failing that, a stub unpickler
that only records state (used below as a fallback).

Licence: Ultralytics YOLOv8 is AGPL-3.0; the Bingsu/adetailer weights (face/hand/person YOLOv8 detectors) carry their own
licence (see the Hugging Face model card, Bingsu/adetailer: AGPL-3.0). Converted files inherit those terms.
"""
import sys, json, pickle, zipfile, io
import torch
from safetensors.torch import save_file


def load_model_state(path):
    """Returns (state_dict, yaml dict, names dict)."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ck.get("ema") or ck["model"]
        return m.state_dict(), dict(getattr(m, "yaml", {}) or {}), dict(getattr(m, "names", {}) or {})
    except ModuleNotFoundError:
        pass

    class Stub:  # records __dict__ state of any unknown class
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): self.__dict__.update(s if isinstance(s, dict) else {"_state": s})

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("ultralytics") or module.startswith("__main__"):
                return type(name, (Stub,), {})
            return super().find_class(module, name)

    class P:  # fake pickle module for torch.load
        Unpickler = U
        load = staticmethod(lambda f, **k: U(f, **k).load())
        __name__ = "pickle"
        UnpicklingError = pickle.UnpicklingError
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=P)
    m = ck.get("ema") or ck["model"]
    sd = {}
    def walk(mod, prefix):
        for k, v in mod.__dict__.get("_parameters", {}).items():
            if v is not None: sd[prefix + k] = v
        for k, v in mod.__dict__.get("_buffers", {}).items():
            if v is not None: sd[prefix + k] = v
        for k, v in mod.__dict__.get("_modules", {}).items():
            if v is not None: walk(v, prefix + k + ".")
    walk(m, "")
    return sd, dict(m.__dict__.get("yaml", {})), dict(m.__dict__.get("names", {}))


for path in sys.argv[1:]:
    sd, yaml, names = load_model_state(path)
    out = {}
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"): continue
        out[k] = v.detach().to(torch.float32).contiguous()
    dst = path.rsplit(".", 1)[0] + ".safetensors"
    save_file(out, dst, metadata={"format": "pt", "source": path.replace("\\", "/").split("/")[-1], "task": "segment" if any(k.startswith("model.22.proto") for k in out) else "detect"})
    with open(dst + ".json", "w", encoding="utf-8") as f:
        json.dump({"names": {str(k): v for k, v in names.items()}, "yaml": yaml}, f, indent=1, default=str)
    print(dst, len(out), "tensors")
