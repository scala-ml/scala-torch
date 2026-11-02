# Maven Central publication

## Current state

The 0.1.0 release is **prepared locally, not published**. The build generates seven Maven modules under `com.github.kmizu`. A source-independent consumer has resolved all seven staged jars and executed CPU autodiff and public CLIP-to-U-Net inference. No Central upload, release tag or GitHub repository creation has occurred.

Publication is authorized by the user for a coherent milestone. Remaining required inputs are the project license, source repository location, and access to the existing Central publisher credentials/PGP signing key. Those choices are pending; the current POMs intentionally do not invent license, homepage or SCM metadata. They are not yet Central-ready. The local environment has neither Central environment credentials nor a secret GPG key. GitHub CLI authentication is available, but does not grant access to Central credentials or recover GitHub Actions secret values.

## Coordinates

All modules use Scala 3 binary suffix `_3`, version 0.1.0 and Java 25 class files:

|sbt project|Artifact|Purpose|
|---|---|---|
|core|scala-torch-core_3|Tensors, layouts and reference operations|
|cpu|scala-torch-cpu_3|numscala CPU execution|
|ffm|scala-torch-ffm_3|Java FFM CUDA operations and resident buffers|
|autograd|scala-torch-autograd_3|CPU automatic differentiation|
|tensorio|scala-torch-io_3|Safetensors readers|
|nn|scala-torch-nn_3|Modules, optimizers and checkpoints|
|models|scala-torch-models_3|Llama, CLIP and U-Net implementations|

The root is aggregate-only and skips publication. Test jars, test fixtures and model weights are not shipped. `scala-torch-models` transitively includes the other modules; a tensor-only consumer can depend on `scala-torch-core`. Dependency-only test scopes do not become compile dependencies in consumer POMs.

After Central publication is verified, the typical dependency will be:

```scala
libraryDependencies += "com.github.kmizu" %% "scala-torch-models" % "0.1.0"
```

## Reproduce local artifact verification

From the repository root, with Java 25 and sbt 2.0.10:

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch publish
```

`publishTo` is explicitly the local staging repository, so this command writes unsigned files beneath `target/sona-staging`; it does not contact Central to upload artifacts. Each module produces its POM, binary jar, sources jar and Javadoc jar, with MD5/SHA1 checksums. The build uses `sbt-pgp` 2.3.2 for the eventual signed step.

From `examples/consumer`, substitute an absolute staging directory:

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false \
  -Dscalatorch.repository=/absolute/path/to/scala-torch/target/sona-staging \
  -batch 'runMain ArtifactSmoke'
```

To include public diffusion-model interoperability, supply the pinned local directories documented in [CLIP](research/clip-text.md) and [U-Net](research/pretrained-unet.md):

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false \
  -Dscalatorch.repository=/absolute/path/to/scala-torch/target/sona-staging \
  -batch 'runMain ArtifactSmoke /absolute/path/to/text_encoder /absolute/path/to/unet'
```

The consumer has no `dependsOn` source-build reference. Its classpath must contain staged scala-torch jars plus external Maven dependencies, not the library's compilation output directories.

## Final publication sequence

Complete the pending license and repository metadata first. Include the selected license and required notices in the distribution. Verify namespace publishing rights using the authorized Central account; the existing numscala coordinates establish the intended organization, not proof of current credential authorization.

Configure the existing signing key and Central credentials outside version control. sbt 2 recognizes `SONATYPE_USERNAME`/`SONATYPE_PASSWORD`; local credentials may instead be configured in `~/.config/sbt/2/credentials.sbt`. Signing requires access to the appropriate secret key and any passphrase. For GitHub Actions, provision repository/organization secrets through the account owner; they cannot be read back from GitHub.

Use a clean staging directory for the final signed build, inspect the exact artifacts, and re-run the standalone consumer against them. Record the clean release commit and tag only after the full CPU, public-model and actual-GPU gates pass. Preliminary unsigned staging is not proof of a signed deployment bundle.

The sbt 2 Central flow is `publishSigned` followed by `sonaUpload` (upload for portal review) or `sonaRelease` (upload and publish). The requested release is already authorized; no duplicate permission question is needed once the inputs and verification are complete. A successful upload is not proof of public availability: check deployment status and resolve every module from Maven Central, then run the standalone consumer without the local resolver override.

Authoritative guides: [sbt 2 Central publishing](https://www.scala-sbt.org/2.x/docs/en/recipes/central.html), [Central artifact requirements](https://central.sonatype.org/publish/requirements/), [Central publisher API](https://central.sonatype.org/publish/publish-portal-api/).

## Verified preparation evidence

After the packaging changes, the complete suite passed **312 tests, 0 failures, 0 skips**: core 19, CPU 15, autograd 70, tensorio 17, nn 43, models 113, FFM 35. All public checkpoint and actual-CUDA gates were enabled. The staged outputs contain exactly seven modules and 28 primary artifacts; all generated MD5/SHA1 files match their payloads. All 363 library class files have Java 25 class version 69. Internal dependency versions are uniformly 0.1.0 and POMs contain no extra repositories or snapshot dependencies.

The independent consumer compiled and ran against those staged jars, passing both CPU/autograd and public CLIP-to-U-Net smoke checks. Read-only review found no preparation blockers and verified runtime PTX resources are included. These checks do not establish Central readiness: license/source metadata, signed artifacts, namespace authorization and publication/resolution remain pending.
