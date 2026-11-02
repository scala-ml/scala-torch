# Independent safetensors fixture

`upstream-0.6.2.safetensors` was generated with the official Rust `safetensors = 0.6.2` serializer, not by the Scala reader. Generator source and Cargo.lock are in `io/src/test/fixture-generator/`.

SHA-256: `47d6471bcb64aa0a7793ce84a02de915a7884c9e8ab8caae3c2512e4d83a740c`

From repository root, optionally regenerate with Rust:

```sh
cargo run --locked --manifest-path io/src/test/fixture-generator/Cargo.toml -- io/src/test/resources/safetensors/upstream-0.6.2.safetensors
sha256sum io/src/test/resources/safetensors/upstream-0.6.2.safetensors
```

Ordinary Scala tests use the committed binary and require neither Rust nor Python. The fixture contains generated constant F32/F16/BF16/F64/I64/BOOL and empty tensors; it contains no downloaded model weights. Its `weight` and `bias` give a known Linear result `[24,52]` for input `[1,2,3]`.
