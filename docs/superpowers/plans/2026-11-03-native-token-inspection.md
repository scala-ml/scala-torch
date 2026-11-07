# Native tokenizer inspection

1. Capture NUI token-inspection responses for SD1 and SDXL tokenizer files, including text, ID input, byte fragments and special tokens. Add failing worker tests with model load/unload and invalid input.
2. Expose the active native tokenizer's vocabulary safely and implement the NUI protocol response, including UTF-8 byte grouping, both SDXL streams and textual-inversion added tokens. Advertise the feature only after integration verification.
3. Compare against independent NUI fixtures, run native tokenizer/worker regression tests and an NUI EngineClient integration. Record slow/fast normalization differences explicitly.
