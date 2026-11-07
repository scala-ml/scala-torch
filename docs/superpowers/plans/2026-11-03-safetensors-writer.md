# SafeTensors writing for native NUI model tools

1. Add failing tests for typed tensors, metadata, scalars, empty and strided tensors, half/bfloat precision, invalid inputs, overwrite protection and interrupted writes.
2. Implement bounded, deterministic header/payload writing with typed encodings and caller-controlled cancellation, staging in the output directory before publication.
3. Verify native round trips and load scala-torch-produced archives with the upstream safetensors library. Record that NUI embedding/model tool integration is still required.
