package scalatorch.models

import scalatorch.Shape
import scalatorch.io.*
import java.nio.file.{Files,Path}
import scala.util.Using

/** Converts an SDXL 1.0 Diffusers directory into a standard LDM single-file `.safetensors` checkpoint. */
private[models] object SdxlSingleFileConverter:
  private def unetDiffusersToLdm(name: String): String =
    def resnet(s: String): String = s.replace("norm1.","in_layers.0.")
      .replace("conv1.","in_layers.2.").replace("norm2.","out_layers.0.")
      .replace("conv2.","out_layers.3.").replace("time_emb_proj.","emb_layers.1.")
      .replace("conv_shortcut.","skip_connection.")
    val downResnet = "down_blocks\\.(\\d+)\\.resnets\\.(\\d+)\\.(.+)".r
    val downAttn = "down_blocks\\.(\\d+)\\.attentions\\.(\\d+)\\.(.+)".r
    val downSample = "down_blocks\\.(\\d+)\\.downsamplers\\.0\\.conv\\.(.+)".r
    val midResnet = "mid_block\\.resnets\\.(\\d+)\\.(.+)".r
    val midAttn = "mid_block\\.attentions\\.0\\.(.+)".r
    val upResnet = "up_blocks\\.(\\d+)\\.resnets\\.(\\d+)\\.(.+)".r
    val upAttn = "up_blocks\\.(\\d+)\\.attentions\\.(\\d+)\\.(.+)".r
    val upSample = "up_blocks\\.(\\d+)\\.upsamplers\\.0\\.conv\\.(.+)".r
    val body = name match
      case s"conv_in.$rest" => s"input_blocks.0.0.$rest"
      case downResnet(b,r,rest) => s"input_blocks.${b.toInt*3+r.toInt+1}.0.${resnet(rest)}"
      case downAttn(b,r,rest) => s"input_blocks.${b.toInt*3+r.toInt+1}.1.$rest"
      case downSample(b,rest) => s"input_blocks.${b.toInt*3+3}.0.op.$rest"
      case midResnet(r,rest) => s"middle_block.${r.toInt*2}.${resnet(rest)}"
      case midAttn(rest) => s"middle_block.1.$rest"
      case upResnet(b,r,rest) => s"output_blocks.${b.toInt*3+r.toInt}.0.${resnet(rest)}"
      case upAttn(b,r,rest) => s"output_blocks.${b.toInt*3+r.toInt}.1.$rest"
      case upSample("2",rest) => s"output_blocks.8.1.conv.$rest"
      case upSample(b,rest) => s"output_blocks.${b.toInt*3+2}.2.conv.$rest"
      case other => Vector(
        "time_embedding.linear_1." -> "time_embed.0.",
        "time_embedding.linear_2." -> "time_embed.2.",
        "add_embedding.linear_1." -> "label_emb.0.0.",
        "add_embedding.linear_2." -> "label_emb.0.2.",
        "conv_norm_out." -> "out.0.",
        "conv_out." -> "out.2.").find((a,_) => other.startsWith(a))
        .map((a,b) => b + other.stripPrefix(a)).getOrElse(
          throw IllegalArgumentException(s"unmapped SDXL U-Net Diffusers key: $other"))
    "model.diffusion_model." + body

  private def vaeDiffusersToLdm(name: String): String =
    def resnet(s: String): String =
      if s.startsWith("conv_shortcut.") then "nin_shortcut." + s.stripPrefix("conv_shortcut.") else s
    val downResnet = "encoder\\.down_blocks\\.(\\d+)\\.resnets\\.(\\d+)\\.(.+)".r
    val downSample = "encoder\\.down_blocks\\.(\\d+)\\.downsamplers\\.0\\.conv\\.(.+)".r
    val upResnet = "decoder\\.up_blocks\\.(\\d+)\\.resnets\\.(\\d+)\\.(.+)".r
    val upSample = "decoder\\.up_blocks\\.(\\d+)\\.upsamplers\\.0\\.conv\\.(.+)".r
    val midResnet = "(encoder|decoder)\\.mid_block\\.resnets\\.(\\d+)\\.(.+)".r
    val midAttn = "(encoder|decoder)\\.mid_block\\.attentions\\.0\\.(.+)".r
    val body = name match
      case downResnet(i,j,rest) => s"encoder.down.$i.block.$j.${resnet(rest)}"
      case downSample(i,rest) => s"encoder.down.$i.downsample.conv.$rest"
      case upResnet(i,j,rest) => s"decoder.up.${3-i.toInt}.block.$j.${resnet(rest)}"
      case upSample(i,rest) => s"decoder.up.${3-i.toInt}.upsample.conv.$rest"
      case midResnet(side,i,rest) => s"$side.mid.block_${i.toInt+1}.${resnet(rest)}"
      case midAttn(side,rest) =>
        val mapped = Vector(
          "group_norm." -> "norm.",
          "to_q." -> "q.", "query." -> "q.",
          "to_k." -> "k.", "key." -> "k.",
          "to_v." -> "v.", "value." -> "v.",
          "to_out.0." -> "proj_out.", "proj_attn." -> "proj_out.").find((a,_) => rest.startsWith(a))
          .map((a,b) => b + rest.stripPrefix(a)).getOrElse(
            throw IllegalArgumentException(s"unmapped VAE mid attention key: $name"))
        s"$side.mid.attn_1.$mapped"
      case other => other.replace("encoder.conv_norm_out.","encoder.norm_out.")
        .replace("decoder.conv_norm_out.","decoder.norm_out.")
    "first_stage_model." + body

  def ensureSingleFile(diffusersDir: Path, output: Path): Path =
    if Files.isRegularFile(output) && Files.size(output) > 6000000000L then return output
    Files.createDirectories(output.getParent)
    val clipL = SafeTensors.open(diffusersDir.resolve("text_encoder").resolve("model.safetensors"))
    val clipG = SafeTensors.open(diffusersDir.resolve("text_encoder_2").resolve("model.safetensors"))
    val unet = SafeTensors.open(diffusersDir.resolve("unet").resolve("diffusion_pytorch_model.safetensors"))
    val vae = SafeTensors.open(diffusersDir.resolve("vae").resolve("diffusion_pytorch_model.safetensors"))
    try
      val plans = Map.newBuilder[String,SafeTensorPlan]
      def copyRaw(archive: SafeTensors, source: String, target: String): Unit =
        val info = archive.tensors(source)
        plans += target -> SafeTensorPlan(info.dtype, info.shape, () =>
          SafeTensorWrite.Raw(info.dtype, info.shape, archive.readRaw(source)))

      clipL.tensors.keysIterator.foreach { key =>
        copyRaw(clipL, key, "conditioner.embedders.0.transformer." + key)
      }

      val prefixG = "conditioner.embedders.1.model."
      copyRaw(clipG, "text_model.embeddings.token_embedding.weight", prefixG + "token_embedding.weight")
      copyRaw(clipG, "text_model.embeddings.position_embedding.weight", prefixG + "positional_embedding")
      copyRaw(clipG, "text_model.final_layer_norm.weight", prefixG + "ln_final.weight")
      copyRaw(clipG, "text_model.final_layer_norm.bias", prefixG + "ln_final.bias")

      val projInfo = clipG.tensors("text_projection.weight")
      val projShape = Shape.from(Seq(projInfo.shape.dims(1), projInfo.shape.dims(0)))
      plans += (prefixG + "text_projection") -> SafeTensorPlan(projInfo.dtype, projShape, () => {
        val raw = clipG.readRaw("text_projection.weight")
        val rows = projInfo.shape.dims(0).toInt
        val cols = projInfo.shape.dims(1).toInt
        val bpe = projInfo.dtype.bytes
        val out = new Array[Byte](raw.length)
        var r = 0
        while r < rows do
          var c = 0
          while c < cols do
            System.arraycopy(raw, (r * cols + c) * bpe, out, (c * rows + r) * bpe, bpe)
            c += 1
          r += 1
        SafeTensorWrite.Raw(projInfo.dtype, projShape, out)
      })

      for i <- 0 until 32 do
        val dBase = s"text_model.encoder.layers.$i."
        val lBase = s"${prefixG}transformer.resblocks.$i."
        Vector(
          "layer_norm1.weight" -> "ln_1.weight", "layer_norm1.bias" -> "ln_1.bias",
          "layer_norm2.weight" -> "ln_2.weight", "layer_norm2.bias" -> "ln_2.bias",
          "self_attn.out_proj.weight" -> "attn.out_proj.weight", "self_attn.out_proj.bias" -> "attn.out_proj.bias",
          "mlp.fc1.weight" -> "mlp.c_fc.weight", "mlp.fc1.bias" -> "mlp.c_fc.bias",
          "mlp.fc2.weight" -> "mlp.c_proj.weight", "mlp.fc2.bias" -> "mlp.c_proj.bias"
        ).foreach((from,to) => copyRaw(clipG, dBase + from, lBase + to))

        for suffix <- Seq("weight","bias") do
          val qKey = s"${dBase}self_attn.q_proj.$suffix"
          val kKey = s"${dBase}self_attn.k_proj.$suffix"
          val vKey = s"${dBase}self_attn.v_proj.$suffix"
          val qInfo = clipG.tensors(qKey)
          val fusedShape = Shape.from((qInfo.shape.dims.head * 3L) +: qInfo.shape.dims.tail)
          plans += s"${lBase}attn.in_proj_$suffix" -> SafeTensorPlan(qInfo.dtype, fusedShape, () => {
            val q = clipG.readRaw(qKey)
            val k = clipG.readRaw(kKey)
            val v = clipG.readRaw(vKey)
            val out = new Array[Byte](q.length + k.length + v.length)
            System.arraycopy(q, 0, out, 0, q.length)
            System.arraycopy(k, 0, out, q.length, k.length)
            System.arraycopy(v, 0, out, q.length + k.length, v.length)
            SafeTensorWrite.Raw(qInfo.dtype, fusedShape, out)
          })

      unet.tensors.keysIterator.foreach { key =>
        copyRaw(unet, key, unetDiffusersToLdm(key))
      }
      vae.tensors.keysIterator.foreach { key =>
        copyRaw(vae, key, vaeDiffusersToLdm(key))
      }

      SafeTensorsWriter.writePlanned(output, plans.result(), overwrite = true)
      output
    finally
      try vae.close() finally
        try unet.close() finally
          try clipG.close() finally clipL.close()
