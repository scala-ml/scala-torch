# OpenCV cubic compatibility for Canny

1. Capture OpenCV 4.11 cubic resize output for varied sizes, border patterns and fractional ratios. Add a failing direct image test and tighten the NUI Canny fixture.
2. Implement a bounded, cancellation-aware OpenCV-style cubic resampler with replicated edges and delayed rounding, leaving Pillow bicubic unchanged. Compare intermediate pixels and final Canny output.
3. Run the reference suite, ControlNet image and NUI preprocess regressions, then rebuild the worker; document any remaining differences.
