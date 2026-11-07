package scalatorch.models.worker

import munit.FunSuite
import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.models.*
import scalatorch.nn.*
import java.awt.image.BufferedImage
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import NativeJson.*

class LycorisPixelArtSubmissionSuite extends FunSuite:
  private def writeSafetensors(dir: Path, name: String, entries: Vector[(String, Vector[Long], Array[Float])]): Path =
    var offset = 0
    val header = entries.map { (k, dims, values) =>
      val begin = offset
      offset += values.length * 4
      s"\"$k\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
    }.mkString("{", ",", "}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data = ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
    entries.foreach((_, _, values) => values.foreach(data.putFloat))
    val path = dir.resolve(name)
    Files.write(
      path,
      ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array() ++ header ++ data.array()
    )
    path

  test("DiffusionLoRA loads LyCORIS LoHa, LoKr, IA3, DoRA, full diff, and CP-LoCon") {
    val dir = Files.createTempDirectory("lycoris-test")
    val linearWeight = Parameter(Tensor.fromFloats(Array(1f, 0f, 0f, 1f), Shape.from(Seq(2L, 2L))), false)
    val convWeight = Parameter(Tensor.fromFloats(new Array[Float](2 * 1 * 3 * 3), Shape.from(Seq(2L, 1L, 3L, 3L))), false)
    val targets = Map(
      "unet.down_blocks.0.attentions.0.to_q" -> linearWeight,
      "unet.down_blocks.0.resnets.0.conv1" -> convWeight
    )
    val input2d = Variable(Tensor.fromFloats(Array(1f, 2f), Shape.from(Seq(1L, 2L))))
    val linearLayer = Linear(linearWeight, None)

    // 1. LoHa with lycoris_unet_ prefix
    val lohaPath = writeSafetensors(
      dir,
      "loha.safetensors",
      Vector(
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.hada_w1_a", Vector(2L, 1L), Array(1f, 2f)),
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.hada_w1_b", Vector(1L, 2L), Array(3f, 4f)),
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.hada_w2_a", Vector(2L, 1L), Array(2f, 1f)),
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.hada_w2_b", Vector(1L, 2L), Array(1f, 2f)),
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.alpha", Vector.empty, Array(1f))
      )
    )
    // w1 = [[3, 4], [6, 8]], w2 = [[2, 4], [1, 2]] -> w1*w2 = [[6, 16], [6, 16]]
    // base + delta = [[7, 16], [6, 17]]; for x = [1, 2], x @ W^T = [7 + 32, 6 + 34] = [39, 40]
    val lohaAdapters = DiffusionLoRA.load(lohaPath, targets)
    val lohaOut = LinearAdapterScope.withAdapters(lohaAdapters)(linearLayer(input2d).data.toFloatArray.toVector)
    assertEquals(lohaOut, Vector(39f, 40f))

    // 2. LoKr with lycoris_unet_ prefix
    val lokrPath = writeSafetensors(
      dir,
      "lokr.safetensors",
      Vector(
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.lokr_w1", Vector(2L, 1L), Array(2f, 3f)),
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.lokr_w2_a", Vector(1L, 1L), Array(1f)),
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.lokr_w2_b", Vector(1L, 2L), Array(4f, 5f)),
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.alpha", Vector.empty, Array(1f))
      )
    )
    // w1 = [[2], [3]], w2 = [[4, 5]] -> kron(w1, w2) = [[8, 10], [12, 15]]
    // base + delta = [[9, 10], [12, 16]]; for x = [1, 2], x @ W^T = [9 + 20, 12 + 32] = [29, 44]
    val lokrAdapters = DiffusionLoRA.load(lokrPath, targets)
    val lokrOut = LinearAdapterScope.withAdapters(lokrAdapters)(linearLayer(input2d).data.toFloatArray.toVector)
    assertEquals(lokrOut, Vector(29f, 44f))

    // 3. IA3 (output scaling and input scaling)
    val ia3Path = writeSafetensors(
      dir,
      "ia3.safetensors",
      Vector(
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.ia3_w", Vector(2L), Array(0.5f, 2.0f)),
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.on_input", Vector(1L), Array(1.0f))
      )
    )
    // base = I_2, on_input scales cols by [0.5, 2.0] -> delta = diag(0.5, 2.0) -> total = diag(1.5, 3.0)
    // for x = [1, 2] -> [1.5, 6.0]
    val ia3Adapters = DiffusionLoRA.load(ia3Path, targets)
    val ia3Out = LinearAdapterScope.withAdapters(ia3Adapters)(linearLayer(input2d).data.toFloatArray.toVector)
    assertEquals(ia3Out, Vector(1.5f, 6.0f))

    // 4. Full diff
    val diffPath = writeSafetensors(
      dir,
      "diff.safetensors",
      Vector(
        ("lycoris_unet_down_blocks_0_attentions_0_to_q.diff", Vector(2L, 2L), Array(1f, 2f, 3f, 4f))
      )
    )
    val diffAdapters = DiffusionLoRA.load(diffPath, targets)
    val diffOut = LinearAdapterScope.withAdapters(diffAdapters)(linearLayer(input2d).data.toFloatArray.toVector)
    assertEquals(diffOut, Vector(6f, 13f))

    // 5. DoRA (dora_scale)
    val doraPath = writeSafetensors(
      dir,
      "dora.safetensors",
      Vector(
        ("lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight", Vector(1L, 2L), Array(0f, 0f)),
        ("lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight", Vector(2L, 1L), Array(0f, 0f)),
        ("lora_unet_down_blocks_0_attentions_0_to_q.dora_scale", Vector(2L), Array(3f, 4f))
      )
    )
    // merged = I_2, row norms = 1, scaled by dora_scale [3, 4] -> W_eff = diag(3, 4) -> for x = [1, 2], out = [3, 8]
    val doraAdapters = DiffusionLoRA.load(doraPath, targets)
    val doraOut = LinearAdapterScope.withAdapters(doraAdapters)(linearLayer(input2d).data.toFloatArray.toVector)
    assert(math.abs(doraOut(0) - 3f) < 1e-5f && math.abs(doraOut(1) - 8f) < 1e-5f)

    // 6. CP-decomposed LoCon (lora_mid.weight)
    val cpPath = writeSafetensors(
      dir,
      "cp-locon.safetensors",
      Vector(
        ("lycoris_unet_down_blocks_0_resnets_0_conv1.lora_down.weight", Vector(1L, 1L, 1L, 1L), Array(2f)),
        ("lycoris_unet_down_blocks_0_resnets_0_conv1.lora_mid.weight", Vector(1L, 1L, 3L, 3L), Array.fill(9)(0.5f)),
        ("lycoris_unet_down_blocks_0_resnets_0_conv1.lora_up.weight", Vector(2L, 1L, 1L, 1L), Array(1f, -1f))
      )
    )
    val cpAdapters = DiffusionLoRA.load(cpPath, targets)
    val convLayer = Conv2d(convWeight, None, Conv2dConfig(padding = (1, 1)))
    val input4d = Variable(Tensor.fromFloats(Array.fill(9)(1f), Shape.from(Seq(1L, 1L, 3L, 3L))))
    val cpOut = LinearAdapterScope.withAdapters(cpAdapters)(convLayer(input4d).data.toFloatArray)
    // Center pixel has 9 ones * (2 * 0.5) = 9.0 for channel 0 and -9.0 for channel 1
    assertEquals(cpOut(4), 9.0f)
    assertEquals(cpOut(9 + 4), -9.0f)
  }

  test("image_tool pixel_art quantizes palette and supports dither, outline, and upscale") {
    val dir = Files.createTempDirectory("pixel-art-test")
    val src = dir.resolve("input.png")
    val img = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 32; x <- 0 until 32 do
      val r = if x < 16 then 240 else 20
      val g = if y < 16 then 220 else 30
      val b = ((x + y) * 4) & 255
      img.setRGB(x, y, (r << 16) | (g << 8) | b)
    ImageIO.write(img, "png", src.toFile)

    val ctx = new NativeWorkContext(() => false, () => true, (_, _) => ())
    val res = NativeCasImageTool.execute(
      Map(
        "tool" -> Str("pixel_art"),
        "image_path" -> Str(src.toString),
        "output_dir" -> Str(dir.toString),
        "pixel_size" -> Number(4),
        "colors" -> Number(4),
        "dither" -> Number(BigDecimal("0.25")),
        "outline" -> Bool(true),
        "upscale" -> Bool(true),
        "seed" -> Number(42)
      ),
      ctx
    )
    val outPath = Path.of(res("path").asInstanceOf[Str].value)
    assert(Files.isRegularFile(outPath))
    val outImg = ImageIO.read(outPath.toFile)
    assertEquals(outImg.getWidth, 32)
    assertEquals(outImg.getHeight, 32)
    val info = res("info").asInstanceOf[Obj].fields
    assertEquals(info("palette"), Number(4))
  }

  test("system_submission_preview and system_submit validate benchmark records and one-use tokens") {
    val record = Obj(
      Map(
        "timestamp" -> Str("2026-11-07T00:00:00Z"),
        "results" -> Arr(Vector(Obj(Map("batch_size" -> Number(1), "iterations_per_second" -> Number(BigDecimal("4.25")))))),
        "system" -> Obj(
          Map(
            "version" -> Obj(Map("app" -> Str("stablediffusion-nui"), "engine" -> Str("scala-torch-0.1.0"))),
            "runtime" -> Obj(Map("pipeline" -> Str("EulerImagePipeline"), "backend" -> Str("cuda"), "dtype" -> Str("fp32")))
          )
        ),
        "config" -> Obj(Map("width" -> Number(512), "height" -> Number(512))),
        "fingerprint" -> Str("abcdef123456"),
        "note" -> Str("unit test")
      )
    )
    val preview = NativeSystemInfo.submissionPreview(
      Map("username" -> Str("tester"), "record" -> record, "include_hostname" -> Bool(false))
    )
    val token = preview("token").asInstanceOf[Str].value
    assert(token.nonEmpty)
    assertEquals(preview("destination"), Str("logs3.papertrailapp.com:32554/udp"))
    val packets = preview("packets").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)
    assertEquals(packets.size, 1)
    assert(packets.head.contains("SDBENCHMARK: tester"))
    assert(packets.head.contains("4.25"))

    var sentHost = ""
    var sentPort = 0
    var sentPackets = Vector.empty[String]
    val submitRes = NativeSystemInfo.submit(
      Map("token" -> Str(token)),
      Some((System.nanoTime(), preview)),
      (pkts, host, port) => {
        sentPackets = pkts
        sentHost = host
        sentPort = port
      }
    )
    assertEquals(submitRes("status"), Str("sent_unconfirmed"))
    assertEquals(sentHost, "logs3.papertrailapp.com")
    assertEquals(sentPort, 32554)
    assertEquals(sentPackets, packets)

    // Invalid or missing token fails
    intercept[IllegalArgumentException] {
      NativeSystemInfo.submit(Map("token" -> Str("wrong-token")), Some((System.nanoTime(), preview)))
    }
    intercept[IllegalArgumentException] {
      NativeSystemInfo.submit(Map("token" -> Str(token)), None)
    }
  }
