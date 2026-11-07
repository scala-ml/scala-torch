package scalatorch.models.worker

import java.nio.file.{Files, Path}
import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*
import scalatorch.nn.*

class TilingCadsHyperTileSuite extends munit.FunSuite:
  test("CircularConvPadding pads X and Y independently with circular or zero padding") {
    // 1x1x2x3 input:
    // [1, 2, 3]
    // [4, 5, 6]
    val x = Tensor.fromFloats(Array(1f, 2f, 3f, 4f, 5f, 6f), Shape.from(Seq(1L, 1L, 2L, 3L)))
    val padXOnly = CircularConvPadding.pad(x, padY = 1, padX = 1, tileX = true, tileY = false)
    assertEquals(padXOnly.shape, Shape.from(Seq(1L, 1L, 4L, 5L)))
    // Row 0 (top pad, tileY=false): all 0
    // Row 1: [3, 1, 2, 3, 1]
    // Row 2: [6, 4, 5, 6, 4]
    // Row 3 (bottom pad, tileY=false): all 0
    assertEquals(
      padXOnly.toFloatArray.toVector,
      Vector(
        0f, 0f, 0f, 0f, 0f,
        3f, 1f, 2f, 3f, 1f,
        6f, 4f, 5f, 6f, 4f,
        0f, 0f, 0f, 0f, 0f
      )
    )

    val padBoth = CircularConvPadding.pad(x, padY = 1, padX = 1, tileX = true, tileY = true)
    assertEquals(
      padBoth.toFloatArray.toVector,
      Vector(
        6f, 4f, 5f, 6f, 4f,
        3f, 1f, 2f, 3f, 1f,
        6f, 4f, 5f, 6f, 4f,
        3f, 1f, 2f, 3f, 1f
      )
    )

    // Verify Conv2d uses CircularConvPadding when active and falls back to zero padding when inactive
    val weight = Parameter(Tensor.fromFloats(Array.fill(9)(1f), Shape.from(Seq(1L, 1L, 3L, 3L))), trainable = false)
    val conv = Conv2d(weight, None, Conv2dConfig(padding = (1, 1)))
    val baseline = conv(Variable(x, requiresGrad = false)).data.toFloatArray.toVector
    val active = CircularConvPadding.withConfig(Some(CircularConvConfig(tileX = true, tileY = true, active = true))) {
      conv(Variable(x, requiresGrad = false)).data.toFloatArray.toVector
    }
    // With full circular padding on 2x3, every 3x3 window wraps all columns and rows:
    // Row sums are (1+2+3)=6 and (4+5+6)=15.
    // Top-row output wraps row 1, row 0, row 1 -> 15 + 6 + 15 = 36 at every column!
    // Bottom-row output wraps row 0, row 1, row 0 -> 6 + 15 + 6 = 27 at every column!
    assertEquals(active, Vector(36f, 36f, 36f, 27f, 27f, 27f))
    assertNotEquals(active, baseline)
  }

  test("CADS gamma schedule and context perturbation match NUI sampling_ext.py") {
    assertEqualsDouble(Cads.gamma(0.0, 0.6, 0.9), 0.0, 1e-9)   // t = 1.0 >= 0.9 -> 0.0
    assertEqualsDouble(Cads.gamma(0.5, 0.6, 0.9), 1.0, 1e-9)   // t = 0.5 <= 0.6 -> 1.0
    assertEqualsDouble(Cads.gamma(0.25, 0.6, 0.9), (0.9 - 0.75) / 0.3, 1e-9)

    val y = Tensor.fromFloats(Array(1f, 2f, 3f, 4f, 5f, 6f), Shape.from(Seq(1L, 2L, 3L)))
    val noise = Tensor.fromFloats(Array(0.5f, -0.5f, 0.25f, -0.25f, 1.0f, -1.0f), Shape.from(Seq(1L, 2L, 3L)))
    // gamma >= 1 returns original tensor unchanged
    assertEquals(Cads.perturb(y, 1.0, 0.25, 1.0, rescale = true, noise).toFloatArray.toVector, y.toFloatArray.toVector)

    // With rescale = true and mixingFactor = 1.0, mean and std match original y
    val rescaled = Cads.perturb(y, 0.5, 0.25, 1.0, rescale = true, noise).toFloatArray
    val origMean = y.toFloatArray.sum / 6.0
    val newMean = rescaled.sum / 6.0
    assertEqualsDouble(newMean, origMean, 1e-5)
    val origStd = math.sqrt(y.toFloatArray.map(v => (v - origMean) * (v - origMean)).sum / 5.0)
    val newStd = math.sqrt(rescaled.map(v => (v - newMean) * (v - newMean)).sum / 5.0)
    assertEqualsDouble(newStd, origStd, 1e-5)
  }

  test("HyperTile divisor, split, tile and untile match NUI hooks.py") {
    assertEquals(HyperTile.randomDivisor(64, 32, 1), 2)
    assertEquals(HyperTile.split(seq = 64 * 64, latentH = 64, latentW = 64, tileSize = 256, swapSize = 1, maxDepth = 0, scaleDepth = false), Some((2, 2, 64, 64)))
    assertEquals(HyperTile.split(seq = 32 * 32, latentH = 32, latentW = 32, tileSize = 256, swapSize = 1, maxDepth = 0, scaleDepth = false), None)

    // Verify tile -> untile is exact identity and groups spatial quadrants properly
    val hs = Tensor.fromFloats((0 until 16).map(_.toFloat).toArray, Shape.from(Seq(1L, 16L, 1L)))
    val tiled = HyperTile.tileTokens(hs, nh = 2, nw = 2, h = 4, w = 4)
    assertEquals(tiled.shape, Shape.from(Seq(4L, 4L, 1L)))
    // Quadrant (0,0) of 4x4 grid has indices (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    assertEquals(tiled.toFloatArray.take(4).toVector, Vector(0f, 1f, 4f, 5f))
    val untiled = HyperTile.untileTokens(tiled, b = 1, nh = 2, nw = 2, h = 4, w = 4)
    assertEquals(untiled.shape, hs.shape)
    assertEquals(untiled.toFloatArray.toVector, hs.toFloatArray.toVector)
  }

  test("NativeDiffusionBackend advertises tiling, cads, and hypertile and executes them in txt2img") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val outDir = Files.createTempDirectory("native-tiling-cads-ht-")
    try
      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      val context = new NativeWorkContext(() => false, () => true, (_, _) => ())
      assert(backend.features.contains("tiling"))
      assert(backend.features.contains("cads"))
      assert(backend.features.contains("hypertile"))

      backend.execute(NativeRequest("1", "load_model", Map("path" -> Str(root.toString)), None), context)

      def runGen(hooks: Map[String, NativeJson], size: Int = 16): Array[Byte] =
        val params = Map[String, NativeJson](
          "prompt" -> Str("a test cat"),
          "negative_prompt" -> Str("blurry"),
          "width" -> Number(size),
          "height" -> Number(size),
          "steps" -> Number(3),
          "seed" -> Number(42),
          "cfg_scale" -> Number(7.5),
          "sampler_name" -> Str("Euler"),
          "scheduler" -> Str("Uniform"),
          "hooks" -> Obj(hooks)
        )
        val req = Map[String, NativeJson](
          "params" -> Obj(params),
          "output_dir" -> Str(outDir.toString),
          "preview_every" -> Number(0),
          "preview_method" -> Str("none")
        )
        val res = backend.execute(NativeRequest("2", "txt2img", req, None), context)
        val imgObj = res("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
        val imgPath = Path.of(imgObj("path").asInstanceOf[Str].value)
        Files.readAllBytes(imgPath)

      val base16 = runGen(Map.empty, size = 16)
      val tiled16 = runGen(Map("tiling" -> Obj(Map("enabled" -> Bool(true), "tile_x" -> Bool(true), "tile_y" -> Bool(false)))), size = 16)
      assert(!java.util.Arrays.equals(base16, tiled16), "expected asymmetric tiling to alter generated image")

      val cads16 = runGen(Map("cads" -> Obj(Map("enabled" -> Bool(true), "tau1" -> Number(BigDecimal("0.1")), "tau2" -> Number(BigDecimal("0.9")), "noise_scale" -> Number(BigDecimal("0.5"))))), size = 16)
      assert(!java.util.Arrays.equals(base16, cads16), "expected CADS to alter generated image")

      val ht16 = runGen(Map("hypertile" -> Obj(Map("enabled" -> Bool(true), "tile_size" -> Number(8), "max_depth" -> Number(2)))), size = 16)
      assert(!java.util.Arrays.equals(base16, ht16), "expected HyperTile to alter generated image when tile_size < image size")
    finally
      val files = Files.walk(outDir)
      try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => { Files.delete(p); () })
      finally files.close()
  }
