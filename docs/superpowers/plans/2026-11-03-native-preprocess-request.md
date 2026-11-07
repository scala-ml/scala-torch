# Native ControlNet preprocess request

1. Capture NUI reference outputs for regular and Extras preprocessing, including source-size and processor-resolution behavior, pixel-perfect Canny, parameter metadata and warning notes. Add failing model-free worker tests.
2. Route NUI `preprocess` to reusable native ControlNet image operations, preserve the standalone PNG text metadata and source parameters, and publish atomically with cancellation.
3. Verify reference pixels, metadata, invalid requests and cancellation; run NUI Generator/EngineClient integration and worker regressions. Record remaining detector and numerical limitations.
