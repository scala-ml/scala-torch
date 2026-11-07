package scalatorch.models

import java.io.RandomAccessFile
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scalatorch.io.SafeTensorSet
import scala.util.Using

class SingleFileSd1InpaintSuite extends munit.FunSuite:
  private def withCheckpoint(channels: Int)(f: SafeTensorSet => Unit): Unit =
    val tensors = Vector(
      "model.diffusion_model.input_blocks.0.0.weight" -> Vector(320,channels,3,3),
      "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight" -> Vector(320,768),
      "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight" -> Vector(49408,768),
      "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight" -> Vector(77,768),
      "first_stage_model.encoder.conv_in.weight" -> Vector(128,3,3,3),
      "first_stage_model.fixture" -> Vector(1))
    var offset = 0L
    val entries = tensors.map { (name,shape) =>
      val start = offset
      offset += shape.map(_.toLong).product * 2
      s"\"$name\":{\"dtype\":\"F16\",\"shape\":[${shape.mkString(",")}],\"data_offsets\":[$start,$offset]}"
    }
    val json = s"{${entries.mkString(",")}}"
    val header = (json + (" " * ((8 - json.length % 8) % 8))).getBytes(StandardCharsets.UTF_8)
    val path = Files.createTempFile("sd1-inpaint-signature-", ".safetensors")
    try
      Using.resource(new RandomAccessFile(path.toFile,"rw")) { file =>
        file.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array())
        file.write(header)
        file.setLength(8L + header.length + offset)
      }
      Using.resource(SafeTensorSet.openFile(path))(f)
    finally Files.deleteIfExists(path)

  test("accepts both standard and nine-channel SD1 checkpoint signatures") {
    for channels <- Seq(4,9) do withCheckpoint(channels) { archive =>
      assertEquals(SingleFileSd1Config.unetConfigFor(archive).inChannels,channels)
      assertEquals(SingleFileSd1Config.unetConfigForTensors(archive.tensors).inChannels,channels)
    }
  }

  test("rejects unsupported SD1 input channel counts") {
    withCheckpoint(5) { archive =>
      intercept[IllegalArgumentException](SingleFileSd1Config.inspect(archive))
      intercept[IllegalArgumentException](SingleFileSd1Config.inspectTensors(archive.tensors))
    }
  }

