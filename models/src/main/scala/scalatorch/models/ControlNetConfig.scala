package scalatorch.models

import com.fasterxml.jackson.core.{JsonFactory,JsonToken,StreamReadConstraints,StreamReadFeature}
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import scalatorch.nn.{Module,Parameter}
import scala.collection.mutable
import scala.util.Using

/** The supported image-conditioning settings around a standard Diffusers SD1 ControlNet U-Net config. */
final case class ControlNetConfig(unet: UNetConfig,conditioningChannels: Int,
    conditioningEmbeddingChannels: Vector[Int],channelOrder: String,globalPoolConditions: Boolean):
  require(unet.inChannels==4 && unet.outChannels==4,
    "native ControlNet requires a four-channel U-Net")
  require(conditioningChannels==3,"native ControlNet requires RGB control images")
  require(conditioningEmbeddingChannels.size==4 && conditioningEmbeddingChannels.forall(n=>n>=1 && n<=4096),
    "ControlNet conditioning embedding requires four bounded channel stages")
  require(channelOrder=="rgb" || channelOrder=="bgr","unsupported ControlNet conditioning channel order")
  require(!globalPoolConditions,"global-pool ControlNet checkpoints are unsupported")

  def requireCompatible(base: UNetConfig): Unit=
    val same=(base.inChannels==unet.inChannels || base.inChannels==2*unet.inChannels+1) &&
      unet.outChannels==base.outChannels &&
      unet.channels==base.channels && unet.downAttention==base.downAttention &&
      unet.layersPerBlock==base.layersPerBlock && unet.contextWidth==base.contextWidth &&
      unet.groups==base.groups && unet.epsilon==base.epsilon &&
      unet.downsamplePadding==base.downsamplePadding &&
      unet.timeWidth==base.timeWidth && unet.flipSinToCos==base.flipSinToCos &&
      unet.frequencyShift==base.frequencyShift && unet.midScale==base.midScale &&
      unet.linearProjection==base.linearProjection && unet.additionTimeWidth==base.additionTimeWidth &&
      unet.additionInputWidth==base.additionInputWidth &&
      unet.channels.indices.forall(i=>unet.headsAt(i)==base.headsAt(i) &&
        unet.transformerLayersAt(i)==base.transformerLayersAt(i))
    require(same,"ControlNet U-Net configuration is incompatible with the loaded base model")

object ControlNetConfig:
  given Module[ControlNetConfig] with
    def namedParameters(value: ControlNetConfig): Vector[(String,Parameter)]=Vector.empty

  private val factory: JsonFactory=JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(8).maxStringLength(1048576).build()).build()
  private val ignored=Set("conditioning_channels","conditioning_embedding_out_channels",
    "controlnet_conditioning_channel_order","global_pool_conditions")

  def read(path: Path): ControlNetConfig =
    require(Files.isRegularFile(path),s"ControlNet config is not a regular file: $path")
    val bytes=Using.resource(Files.newInputStream(path))(_.readNBytes(1048577))
    require(bytes.length<=1048576,"ControlNet config exceeds 1MiB")
    val json=UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString
    val fields=readControlFields(json)
    val base=UNetConfig.fromJson(json,"ControlNetModel",ignored)
    ControlNetConfig(base,fields.channels,fields.embeddingChannels,fields.order,fields.globalPool)

  private final case class ControlFields(channels: Int,embeddingChannels: Vector[Int],order: String,globalPool: Boolean)
  private def readControlFields(json: String): ControlFields =
    require(json.length<=1048576,"ControlNet config exceeds size limit")
    try Using.resource(factory.createParser(json)) { p =>
      require(p.nextToken()==JsonToken.START_OBJECT,"ControlNet config must be an object")
      var className: Option[String]=None
      var channels=3
      var embedding=Vector(16,32,96,256)
      var order="rgb"
      var globalPool=false
      val seen=mutable.Set.empty[String]
      var fields=0
      while p.nextToken()!=JsonToken.END_OBJECT do
        require(p.currentToken()==JsonToken.FIELD_NAME && fields<128,"invalid ControlNet config field")
        val key=p.currentName()
        require(seen.add(key),s"duplicate ControlNet config field: $key")
        fields+=1
        p.nextToken()
        key match
          case "_class_name" =>
            require(p.currentToken()==JsonToken.VALUE_STRING,"invalid ControlNet class name")
            className=Some(p.getText())
          case "conditioning_channels" =>
            require(p.currentToken()==JsonToken.VALUE_NUMBER_INT,"invalid ControlNet conditioning channel count")
            channels=p.getIntValue()
          case "conditioning_embedding_out_channels" =>
            require(p.currentToken()==JsonToken.START_ARRAY,"invalid ControlNet conditioning embedding channels")
            val out=Vector.newBuilder[Int]
            var count=0
            while p.nextToken()!=JsonToken.END_ARRAY do
              require(p.currentToken()==JsonToken.VALUE_NUMBER_INT && count<8,"invalid ControlNet conditioning embedding channels")
              out+=p.getIntValue();count+=1
            embedding=out.result()
          case "controlnet_conditioning_channel_order" =>
            require(p.currentToken()==JsonToken.VALUE_STRING,"invalid ControlNet channel order")
            order=p.getText()
          case "global_pool_conditions" =>
            require(p.currentToken()==JsonToken.VALUE_TRUE || p.currentToken()==JsonToken.VALUE_FALSE,
              "invalid ControlNet global-pool flag")
            globalPool=p.getBooleanValue()
          case _ => p.skipChildren()
      require(p.nextToken()==null,"trailing ControlNet config JSON")
      require(className.contains("ControlNetModel"),"config is not a Diffusers ControlNetModel")
      ControlFields(channels,embedding,order,globalPool)
    }
    catch case e: IOException => throw IllegalArgumentException("invalid ControlNet config JSON",e)
