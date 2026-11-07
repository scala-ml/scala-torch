# Native NUI CAS image tool

1. Generate fixed source and output pixels from NUI's Pillow/NumPy CAS implementation, including RGBA input, non-integer scale and sharpening extremes. Add a failing worker request test without a loaded model.
2. Add the `image_tool` protocol route and native CAS implementation with bounded image IO, orientation, cancellation and atomic publication. Explicitly reject other image tools until implemented.
3. Verify reference pixel agreement, protocol result, output cleanup on cancelled publication and existing image IO/upscale tests; rebuild the worker.
