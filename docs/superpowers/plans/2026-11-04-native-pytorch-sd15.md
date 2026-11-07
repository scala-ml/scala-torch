# Native PyTorch SD1 checkpoint

Load standard SD1 PyTorch ZIP checkpoints without executing pickle code or invoking Python. Retain strict SD1 signature, shape and per-component tensor checks, transactional model publication, cancellation and existing safetensors behavior. Derive component views from a bounded restricted archive and release decoded arrays as components are consumed. Explicitly reject unsupported checkpoint layouts and dtypes.

1. Inspect the published SD1.5 checkpoint's archive layout and tensor dtypes; add small parser/view/config tests before changing code.
2. Generalize SD1 component name mapping and signature checking to both safetensors and restricted PyTorch tensor sources, without changing existing file behavior.
3. Integrate `.ckpt` selection, VAE reset, model metadata, cancellation and model switching in the native worker.
4. Compare real checkpoint output with the corresponding safetensors model on 4 GB CUDA, then verify NUI Generator and package path as appropriate. Record limits and host-memory observations.
