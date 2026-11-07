package scalatorch.models.worker

import scalatorch.*
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path,StandardOpenOption}

object NativeVaeRecovery:
  def save(input: Tensor[Float],directory: Path): Path=
    require(input.device==Device.Cpu && input.dtype==DType.Float32,"recovery requires CPU Float32 input")
    val values=input.toFloatArray
    val bytes=values.length.toLong*4
    val shape=input.shape.dims.mkString(",")
    val raw=s"""{"__metadata__":{"coordinate":"vae_decoder_input","producer":"scala-torch"},"tensor":{"dtype":"F32","shape":[$shape],"data_offsets":[0,$bytes]}}"""
      .getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val padded=((raw.length+7)/8)*8
    val header=Array.fill[Byte](padded)(32)
    System.arraycopy(raw,0,header,0,raw.length)
    val folder=Files.createDirectories(directory.resolve("vae-recovery"))
    val path=folder.resolve(s"caught-latent-${java.util.UUID.randomUUID()}.safetensors")
    val output=Files.newOutputStream(path,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
    try
      try
        output.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(padded.toLong).array())
        output.write(header)
        val buffer=ByteBuffer.allocate(65536).order(ByteOrder.LITTLE_ENDIAN)
        values.foreach { value =>
          if buffer.remaining()<4 then
            output.write(buffer.array(),0,buffer.position());buffer.clear()
          buffer.putFloat(value)
        }
        if buffer.position()>0 then output.write(buffer.array(),0,buffer.position())
      finally output.close()
      path
    catch
      case error: Throwable =>
        try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
