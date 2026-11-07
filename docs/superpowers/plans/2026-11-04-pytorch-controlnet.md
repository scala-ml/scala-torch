# PyTorch SD1 ControlNet archives

Allow the restricted PyTorch ZIP reader to load standard SD1 ControlNet state dictionaries into the existing strict model loader without invoking Python. Support original and Diffusers key layouts, direct dictionaries and a `state_dict` wrapper; reject unsupported pickle globals, non-tensor entries, mixed layouts, wrong shapes and exhausted resource budgets. Keep existing safetensors and directory behavior.

1. Add small state-dictionary tests for mapping and rejection, and verify existing safetensors tests.
2. Adapt the lazy tensor view to an already-decoded PyTorch tensor map and configure bounded model-sized archive limits.
3. Exercise the original published FP32 Canny `.pth` on the 4 GB GPU, compare against an independently converted safetensors model and verify NUI catalog/Generator integration.
