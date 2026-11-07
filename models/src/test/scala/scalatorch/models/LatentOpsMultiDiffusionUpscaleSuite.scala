package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scalatorch.*
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*

class LatentOpsMultiDiffusionUpscaleSuite extends munit.FunSuite:
  private def writeRrdbSafeTensors(path: Path, blocks: Int = 1): Unit =
    val entries = scala.collection.mutable.ArrayBuffer.empty[(String, Vector[Long], Array[Float])]
    def addConv(prefix: String, inC: Int, outC: Int, identityPass: Boolean = false): Unit =
      val wShape = Vector(outC.toLong, inC.toLong, 3L, 3L)
      val wArr = new Array[Float](outC * inC * 9)
      if identityPass then
        for c <- 0 until math.min(inC, outC) do
          wArr((c * inC + c) * 9 + 4) = 0.25f
      entries += ((s"$prefix.weight", wShape, wArr))
      val bArr = Array.tabulate(outC)(c => if identityPass && c < 3 then 0.05f * (c + 1) else 0f)
      entries += ((s"$prefix.bias", Vector(outC.toLong), bArr))

    addConv("conv_first", 3, 64, identityPass = true)
    for b <- 0 until blocks; d <- 1 to 3 do
      val root = s"RRDB_trunk.$b.RDB$d"
      addConv(s"$root.conv1", 64, 32)
      addConv(s"$root.conv2", 96, 32)
      addConv(s"$root.conv3", 128, 32)
      addConv(s"$root.conv4", 160, 32)
      addConv(s"$root.conv5", 192, 64)
    addConv("trunk_conv", 64, 64, identityPass = true)
    addConv("upconv1", 64, 64, identityPass = true)
    addConv("upconv2", 64, 64, identityPass = true)
    addConv("HRconv", 64, 64, identityPass = true)
    addConv("conv_last", 64, 3, identityPass = true)

    var offset = 0L
    val jsonParts = entries.map { case (name, shape, values) =>
      val byteLen = values.length * 4L
      val part = s""""$name":{"dtype":"F32","shape":[${shape.mkString(",")}],"data_offsets":[$offset,${offset + byteLen}]}"""
      offset += byteLen
      part
    }
    val headerBytes = jsonParts.mkString("{", ",", "}").getBytes(StandardCharsets.UTF_8)
    val totalBytes = 8 + headerBytes.length + offset.toInt
    val buf = ByteBuffer.allocate(totalBytes).order(ByteOrder.LITTLE_ENDIAN)
    buf.putLong(headerBytes.length.toLong)
    buf.put(headerBytes)
    entries.foreach { case (_, _, values) =>
      values.foreach(buf.putFloat)
    }
    Files.write(path, buf.array())

  test("CudaRealEsrganX4Plus.upscaleCpu executes RRDBNet on CPU and NativeHires accepts R-ESRGAN 4x+ with upscaler_paths") {
    val tmpDir = Files.createTempDirectory("rrdb-cpu-test")
    try
      val modelPath = tmpDir.resolve("RealESRGAN_x4plus.safetensors")
      writeRrdbSafeTensors(modelPath, blocks = 1)
      val src = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 4; x <- 0 until 4 do
        src.setRGB(x, y, ((x * 40) << 16) | ((y * 50) << 8) | 100)
      val upscaled = CudaRealEsrganX4Plus.upscaleCpu(src, modelPath)
      assertEquals((upscaled.getWidth, upscaled.getHeight), (16, 16))

      val parsed = NativeHires.parse(
        Some(Obj(Map("enabled" -> Bool(true), "scale" -> Number(2), "upscaler" -> Str("R-ESRGAN 4x+"), "second_pass_steps" -> Number(2)))),
        Map("R-ESRGAN 4x+" -> modelPath.toString)
      )
      assert(parsed.isDefined)
      assertEquals(parsed.get.upscaler, "R-ESRGAN 4x+")
      assertEquals(parsed.get.upscalerPath, Some(modelPath.toString))
    finally
      val stream = Files.walk(tmpDir)
      try stream.sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists(_))
      finally stream.close()
  }

  test("MultiDiffusion tileStarts, tileGrid, and gaussianWeights match Python reference") {
    assertEquals(MultiDiffusion.tileStarts(16, 16, 4), Vector(0))
    assertEquals(MultiDiffusion.tileStarts(24, 16, 4), Vector(0, 8))
    val grid = MultiDiffusion.tileGrid(24, 24, 16, 16, 4)
    assertEquals(grid.size, 4)
    assertEquals(grid.head, (0, 16, 0, 16))
    assertEquals(grid.last, (8, 24, 8, 24))
    val gw = MultiDiffusion.gaussianWeights(8, 8)
    assertEquals(gw.length, 64)
    assert(gw(3 * 8 + 3) > gw(0), "center weight should exceed corner weight")
  }

  test("LatentModifier applies sharpness, contrast, tonemap, combat_cfg_drift, divisive_norm, and extra_noise") {
    val x = Tensor.fromFloats(Array.tabulate(1 * 4 * 8 * 8)(i => math.sin(i * 0.3).toFloat * 2f), Shape.from(Seq(1L, 4L, 8L, 8L)))
    val cfg = LatentModifierConfig(
      sharpnessMultiplier = 0.5,
      sharpnessMethod = "anisotropic",
      contrastMultiplier = 0.2,
      tonemapMultiplier = 0.5,
      tonemapMethod = "reinhard",
      tonemapPercentile = 95.0,
      combatMethod = "subtract",
      combatCfgDrift = 0.3,
      divisiveNormSize = 5,
      divisiveNormMultiplier = 0.2,
      extraNoiseType = "gaussian",
      extraNoiseMultiplier = 0.1,
      extraNoiseLowpass = 50
    )
    val out = LatentModifier.modifyLatents(x, cfg, progress = 0.25, rng = Some(new TorchCpuNormal(123L)))
    assertEquals(out.shape, x.shape)
    val outArr = out.toFloatArray
    assert(outArr.forall(_.isFinite))
    assert(!java.util.Arrays.equals(x.toFloatArray, outArr))
  }

  test("SoftInpainting softMaskThreshold, softInpaintBlend, and compositePixels match Python semantics") {
    assert(math.abs(SoftInpainting.softMaskThreshold(0, 4, 1.0) - 0.75) < 1e-9)
    assert(math.abs(SoftInpainting.softMaskThreshold(3, 4, 1.0) - 0.0) < 1e-9)
    val lat = Tensor.fromFloats(Array.fill(1 * 4 * 2 * 2)(1f), Shape.from(Seq(1L, 4L, 2L, 2L)))
    val orig = Tensor.fromFloats(Array.fill(1 * 4 * 2 * 2)(-1f), Shape.from(Seq(1L, 4L, 2L, 2L)))
    val mask = Tensor.fromFloats(Array(0f, 0.5f, 0.8f, 1f), Shape.from(Seq(1L, 1L, 2L, 2L)))
    val siCfg = SoftInpaintingConfig(
      maskBlendPower = 1.0,
      maskBlendScale = 0.5,
      inpaintDetailPreservation = 4.0,
      compositeMaskInfluence = 0.5,
      compositeDifferenceThreshold = 0.5,
      compositeDifferenceContrast = 2.0
    )
    val blended = SoftInpainting.softInpaintBlend(lat, orig, mask, 1, 4, siCfg).toFloatArray
    assert(blended(0) < blended(3), "higher mask values should retain more generated latent")

    val origImg = Tensor.fromFloats(Array.fill(1 * 3 * 2 * 2)(-0.8f), Shape.from(Seq(1L, 3L, 2L, 2L)))
    val genImg = Tensor.fromFloats(Array.fill(1 * 3 * 2 * 2)(0.8f), Shape.from(Seq(1L, 3L, 2L, 2L)))
    val comp = SoftInpainting.compositePixels(origImg, genImg, mask, siCfg).toFloatArray
    assert(comp.forall(_.isFinite))
    assert(comp(0) < comp(3), "masked pixel should take more from generated image")
  }

  test("MultiDiffusion, LatentModifier, SoftInpainting, NeverOOM, and Hires R-ESRGAN 4x+ execute in NativeDiffusionBackend") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
    val ctx = new NativeWorkContext(() => false, () => true, (_, _) => ())
    backend.execute(NativeRequest("load", "load_model", Map.empty, None), ctx)

    val outDir = Files.createTempDirectory("latent-ops-md-out")
    try
      val rrdbPath = outDir.resolve("RealESRGAN_x4plus.safetensors")
      writeRrdbSafeTensors(rrdbPath, blocks = 1)

      val resTxt = backend.execute(
        NativeRequest(
          "gen-md-lm",
          "txt2img",
          Map(
            "output_dir" -> Str(outDir.toString),
            "preview_every" -> Number(0),
            "preview_method" -> Str("none"),
            "upscaler_paths" -> Obj(Map("R-ESRGAN 4x+" -> Str(rrdbPath.toString))),
            "params" -> Obj(
              Map(
                "prompt" -> Str("a cat"),
                "negative_prompt" -> Str(""),
                "width" -> Number(16),
                "height" -> Number(16),
                "steps" -> Number(2),
                "seed" -> Number(42),
                "cfg_scale" -> Number(5),
                "sampler_name" -> Str("Euler"),
                "scheduler" -> Str("Uniform"),
                "hires" -> Obj(
                  Map(
                    "enabled" -> Bool(true),
                    "scale" -> Number(4),
                    "upscaler" -> Str("R-ESRGAN 4x+"),
                    "second_pass_steps" -> Number(1),
                    "denoising_strength" -> Number(BigDecimal("0.5"))
                  )
                ),
                "hooks" -> Obj(
                  Map(
                    "multidiffusion" -> Obj(
                      Map(
                        "enabled" -> Bool(true),
                        "method" -> Str("Mixture of Diffusers"),
                        "tile_width" -> Number(64),
                        "tile_height" -> Number(64),
                        "tile_overlap" -> Number(8),
                        "tile_batch_size" -> Number(2)
                      )
                    ),
                    "latent_modifier" -> Obj(
                      Map(
                        "enabled" -> Bool(true),
                        "sharpness_multiplier" -> Number(BigDecimal("0.3")),
                        "contrast_multiplier" -> Number(BigDecimal("0.1")),
                        "tonemap_multiplier" -> Number(BigDecimal("0.2"))
                      )
                    ),
                    "neveroom" -> Obj(
                      Map(
                        "enabled" -> Bool(true),
                        "unet_offload" -> Bool(true),
                        "vae_tiling" -> Bool(true)
                      )
                    )
                  )
                )
              )
            )
          ),
          None
        ),
        ctx
      )
      val images = resTxt("images").asInstanceOf[Arr].values
      assertEquals(images.size, 1)
      val imgObj = images.head.asInstanceOf[Obj].fields
      assertEquals(imgObj("width"), Number(64))
      assertEquals(imgObj("height"), Number(64))

      val initImg = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB)
      val maskImg = new BufferedImage(16, 16, BufferedImage.TYPE_BYTE_GRAY)
      for y <- 0 until 16; x <- 0 until 16 do
        initImg.setRGB(x, y, 0x4080c0)
        maskImg.getRaster.setSample(x, y, 0, if x >= 8 then 255 else 0)
      val initPath = outDir.resolve("init.png")
      val maskPath = outDir.resolve("mask.png")
      ImageIO.write(initImg, "png", initPath.toFile)
      ImageIO.write(maskImg, "png", maskPath.toFile)

      val resInpaint = backend.execute(
        NativeRequest(
          "gen-soft-inpaint",
          "img2img",
          Map(
            "output_dir" -> Str(outDir.toString),
            "preview_every" -> Number(0),
            "preview_method" -> Str("none"),
            "params" -> Obj(
              Map(
                "prompt" -> Str("a dog"),
                "negative_prompt" -> Str(""),
                "width" -> Number(16),
                "height" -> Number(16),
                "steps" -> Number(2),
                "seed" -> Number(42),
                "cfg_scale" -> Number(5),
                "sampler_name" -> Str("Euler"),
                "scheduler" -> Str("Uniform"),
                "img2img" -> Obj(
                  Map(
                    "init_image" -> Str(initPath.toString),
                    "mask" -> Str(maskPath.toString),
                    "denoising_strength" -> Number(BigDecimal("0.75"))
                  )
                ),
                "hooks" -> Obj(
                  Map(
                    "soft_inpainting" -> Obj(
                      Map(
                        "enabled" -> Bool(true),
                        "mask_blend_power" -> Number(1),
                        "mask_blend_scale" -> Number(BigDecimal("0.5")),
                        "inpaint_detail_preservation" -> Number(4),
                        "composite_mask_influence" -> Number(BigDecimal("0.2"))
                      )
                    )
                  )
                )
              )
            )
          ),
          None
        ),
        ctx
      )
      assertEquals(resInpaint("images").asInstanceOf[Arr].values.size, 1)
    finally
      val stream = Files.walk(outDir)
      try stream.sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists(_))
      finally stream.close()
  }
