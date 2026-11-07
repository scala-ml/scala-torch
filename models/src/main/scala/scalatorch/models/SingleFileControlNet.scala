package scalatorch.models

import scalatorch.io.{PyTorchTensorArchive,SafeTensorSet}

/** Original and Diffusers SD1 ControlNet tensor views without copying weights. */
private[models] object SingleFileControlNet:
  val config: ControlNetConfig = ControlNetConfig(SingleFileSd1Config.unetConfig,3,Vector(16,32,96,256),"rgb",false)
  def mapName(source: String): String =
    val name=source.stripPrefix("control_model.")
    val input="input_blocks\\.(\\d+)\\.([01])\\.(.+)".r
    val middle="middle_block\\.([012])\\.(.+)".r
    val hint="input_hint_block\\.(\\d+)\\.(weight|bias)".r
    val zero="zero_convs\\.(\\d+)\\.0\\.(weight|bias)".r
    def resnet(s: String): String = s.replace("in_layers.0.","norm1.")
      .replace("in_layers.2.","conv1.").replace("out_layers.0.","norm2.")
      .replace("out_layers.3.","conv2.").replace("emb_layers.1.","time_emb_proj.")
      .replace("skip_connection.","conv_shortcut.")
    name match
      case input("0","0",rest) => s"conv_in.$rest"
      case input(i,"0",rest) if i.toInt>=1 && i.toInt<=11 && rest.startsWith("op.") =>
        s"down_blocks.${(i.toInt-1)/3}.downsamplers.0.conv.${rest.stripPrefix("op.")}"
      case input(i,"0",rest) if i.toInt>=1 && i.toInt<=11 =>
        s"down_blocks.${(i.toInt-1)/3}.resnets.${(i.toInt-1)%3}.${resnet(rest)}"
      case input(i,"1",rest) if i.toInt>=1 && i.toInt<=11 =>
        s"down_blocks.${(i.toInt-1)/3}.attentions.${(i.toInt-1)%3}.$rest"
      case middle("1",rest) => s"mid_block.attentions.0.$rest"
      case middle(i,rest) => s"mid_block.resnets.${i.toInt/2}.${resnet(rest)}"
      case hint("0",param) => s"controlnet_cond_embedding.conv_in.$param"
      case hint("14",param) => s"controlnet_cond_embedding.conv_out.$param"
      case hint(i,param) if Set(2,4,6,8,10,12)(i.toInt) =>
        s"controlnet_cond_embedding.blocks.${i.toInt/2-1}.$param"
      case zero(i,param) if i.toInt<=11 => s"controlnet_down_blocks.$i.$param"
      case s if s.startsWith("middle_block_out.0.") => "controlnet_mid_block."+s.stripPrefix("middle_block_out.0.")
      case s if s.startsWith("time_embed.0.") => "time_embedding.linear_1."+s.stripPrefix("time_embed.0.")
      case s if s.startsWith("time_embed.2.") => "time_embedding.linear_2."+s.stripPrefix("time_embed.2.")
      case _ => throw IllegalArgumentException(s"unsupported ControlNet tensor: $source")

  def view(archive: SafeTensorSet): SingleFileTensorView =
    SingleFileTensorView(archive,mappings(archive.tensors.keys.toVector))

  def pytorchView(value: PyTorchTensorArchive.Value): SingleFileTensorView =
    import PyTorchTensorArchive.*
    val fields=value match
      case DictValue(root) if root.contains("state_dict") => root("state_dict") match
        case DictValue(state) => state
        case _ => throw IllegalArgumentException("ControlNet state_dict must be a tensor dictionary")
      case DictValue(root) => root
      case _ => throw IllegalArgumentException("ControlNet archive must be a tensor dictionary")
    val tensors=fields.map { (name,entry) => entry match
      case TensorValue(tensor) => name->tensor
      case _ => throw IllegalArgumentException(s"ControlNet archive value is not a tensor: $name")
    }
    SingleFileTensorView.pytorch(tensors,mappings(tensors.keys.toVector))

  private def mappings(names: Vector[String]): Vector[(String,String)] =
    require(names.nonEmpty,"empty ControlNet checkpoint")
    if names.contains("time_embedding.linear_1.weight") then
      val originalPrefixes=Vector("control_model.","input_blocks.","middle_block.","middle_block_out.",
        "input_hint_block.","zero_convs.","time_embed.")
      require(!names.exists(name => originalPrefixes.exists(name.startsWith)),"mixed ControlNet tensor formats are unsupported")
      names.map(name => name->name)
    else
      require(names.forall(_.startsWith("control_model.")) || names.forall(name => !name.startsWith("control_model.")),
        "mixed ControlNet tensor prefixes are unsupported")
      names.map(name => mapName(name)->name)
