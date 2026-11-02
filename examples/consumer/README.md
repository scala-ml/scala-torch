# Standalone artifact consumer

This is an independent sbt 2 build using Maven coordinates, with no source dependency on the parent project. It requires Java 25 and verifies CPU autodiff. Optional arguments additionally load the pinned tiny-sd-pipe CLIP and U-Net weights and run their composed forward pass.

Until 0.1.0 is published, follow the local staging commands in [PUBLISHING.md](../../docs/PUBLISHING.md). After release, omit `-Dscalatorch.repository` to resolve from Maven Central. Run `show Compile / fullClasspath` to inspect the exact jars used.
