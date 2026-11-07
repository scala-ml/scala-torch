# NUI CLIP fallback text normalization

1. Capture tokenizer-inspection references from NUI's installed Transformers without optional ftfy, including adjacent CJK, controls, special tokens and Greek casing. Reproduce the mismatch in native worker tests.
2. Add an explicit NUI fallback normalization profile while preserving the public CLIP fast profile; use it for native generation and inspection, including added-token copies and both SDXL encoders.
3. Verify fixtures, fast profile regressions, generation token chunks, NUI integration and rebuild the worker. Document that optional ftfy behavior remains separate.
