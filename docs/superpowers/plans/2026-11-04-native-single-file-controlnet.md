# Native SD1 single-file ControlNet

Support standard original SD1 ControlNet safetensors through the existing worker and NUI catalog. Map original LDM names into the existing strict Diffusers loader, retain all shape and unused-weight checks, and infer only the standard SD1 architecture. Keep the Diffusers-directory path unchanged. Cache by file identity and invalidate on changes. Reject unsupported layouts explicitly.

1. Test mapping and rejection of unsupported original names.
2. Implement lazy mapped views and file-aware loading / cache / request validation.
3. Compare a real v1.1 Canny single-file against its independently converted Diffusers weights and exercise real CUDA generation, including the NUI request path.
