package scalatorch.models

import com.fasterxml.jackson.core.{JsonFactory,JsonToken,StreamReadConstraints,StreamReadFeature}
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import scala.collection.mutable
import scala.util.Using

/** Supported additive-conditioning Diffusers U-Net configuration. Bounds limit graph construction. */
final case class UNetConfig(inChannels: Int,outChannels: Int,channels: Vector[Int],
    downAttention: Vector[Boolean],upAttention: Vector[Boolean],layersPerBlock: Int,heads: Int,
    contextWidth: Int,groups: Int,epsilon: Double,transformerLayers: Int,downsamplePadding: Int,
    timeWidth: Int,flipSinToCos: Boolean,frequencyShift: Double,midScale: Double):
  require(channels.nonEmpty && channels.size<=8 && downAttention.size==channels.size && upAttention.size==channels.size,
    "invalid U-Net resolution stages")
  require((channels++Vector(inChannels,outChannels,contextWidth,timeWidth)).forall(n => n>0 && n<=16384),
    "U-Net dimensions must be in 1..16384")
  require(layersPerBlock>0 && layersPerBlock<=16 && transformerLayers>0 && transformerLayers<=16,"invalid U-Net depth")
  require(groups>0 && heads>0 && channels.forall(c => c%groups==0 && c%heads==0),"invalid U-Net groups/heads")
  require(epsilon.isFinite && epsilon>0 && frequencyShift.isFinite && channels.head/2-frequencyShift>0,
    "invalid U-Net norm/timestep settings")
  require(channels.head>=2 && (downsamplePadding==0 || downsamplePadding==1),"invalid U-Net sampling settings")
  require(midScale.isFinite && midScale>0 && (1/midScale).toFloat.isFinite && (1/midScale).toFloat>0,"invalid U-Net middle scale")

object UNetConfig:
  private enum Value:
    case Number(n: BigDecimal)
    case Text(s: String)
    case Bool(b: Boolean)
    case Array(values: Vector[Value])
    case Null
  import Value.*
  private val factory: JsonFactory=JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(8).maxStringLength(1048576).build()).build()
  def read(path: Path): UNetConfig = Using.resource(Files.newInputStream(path)) { stream =>
    val bytes=stream.readNBytes(1048577)
    require(bytes.length<=1048576,"U-Net config exceeds 1MiB")
    fromJson(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString)
  }
  def fromJson(json: String): UNetConfig =
    require(json.length<=1048576,"U-Net config exceeds size limit")
    try Using.resource(factory.createParser(json)) { p =>
      def value(arrayAllowed: Boolean=true): Value = p.currentToken() match
        case JsonToken.VALUE_NUMBER_INT | JsonToken.VALUE_NUMBER_FLOAT => Number(BigDecimal(p.getDecimalValue()))
        case JsonToken.VALUE_STRING => Text(p.getText())
        case JsonToken.VALUE_TRUE => Bool(true)
        case JsonToken.VALUE_FALSE => Bool(false)
        case JsonToken.VALUE_NULL => Null
        case JsonToken.START_ARRAY if arrayAllowed =>
          val out=Vector.newBuilder[Value]
          var count=0
          while p.nextToken()!=JsonToken.END_ARRAY do
            require(count<32,"U-Net config array too long")
            out+=value(false);count+=1
          Array(out.result())
        case _ => throw IllegalArgumentException("unsupported U-Net config value")
      require(p.nextToken()==JsonToken.START_OBJECT,"U-Net config must be an object")
      val fields=mutable.Map.empty[String,Value]
      while p.nextToken()!=JsonToken.END_OBJECT do
        require(p.currentToken()==JsonToken.FIELD_NAME && fields.size<128,"invalid U-Net config field")
        val key=p.currentName();p.nextToken();fields(key)=value()
      require(p.nextToken()==null,"trailing U-Net config JSON")
      def take(key: String,default: Value): Value=fields.remove(key).getOrElse(default)
      def exact(key: String,expected: Value): Unit=require(take(key,expected)==expected,s"unsupported U-Net option: $key")
      def integerValue(v: Value,key: String): Int=v match
        case Number(n) if n.isWhole && n>=0 && n<=16384 => n.toInt
        case _ => throw IllegalArgumentException(s"invalid U-Net integer: $key")
      def integer(key: String,default: Int): Int=integerValue(take(key,Number(BigDecimal(default))),key)
      def number(key: String,default: Double): Double=take(key,Number(BigDecimal(default))) match
        case Number(n) if n.toDouble.isFinite => n.toDouble
        case _ => throw IllegalArgumentException(s"invalid U-Net number: $key")
      def boolean(key: String,default: Boolean): Boolean=take(key,Bool(default)) match
        case Bool(b) => b
        case _ => throw IllegalArgumentException(s"invalid U-Net boolean: $key")
      def array(key: String): Vector[Value]=take(key,Null) match
        case Array(v) => v
        case _ => throw IllegalArgumentException(s"missing U-Net array: $key")
      exact("_class_name",Text("UNet2DConditionModel"))
      take("_diffusers_version",Null) match
        case Null | Text(_) => ()
        case _ => throw IllegalArgumentException("invalid diffusers version metadata")
      for key <- Seq("addition_embed_type","addition_time_embed_dim","class_embed_type","cross_attention_norm",
        "encoder_hid_dim","encoder_hid_dim_type","num_attention_heads","num_class_embeds",
        "projection_class_embeddings_input_dim","reverse_transformer_layers_per_block","time_cond_proj_dim",
        "time_embedding_act_fn","timestep_post_act") do exact(key,Null)
      for key <- Seq("center_input_sample","class_embeddings_concat","dual_cross_attention","only_cross_attention",
        "resnet_skip_time_act","upcast_attention","use_linear_projection") do exact(key,Bool(false))
      require(Set[Value](Null,Bool(false))(take("mid_block_only_cross_attention",Null)),"unsupported middle cross attention")
      for (key,v) <- Seq("act_fn"->"silu","attention_type"->"default","mid_block_type"->"UNetMidBlock2DCrossAttn",
        "resnet_time_scale_shift"->"default","time_embedding_type"->"positional") do exact(key,Text(v))
      exact("conv_in_kernel",Number(3));exact("conv_out_kernel",Number(3))
      exact("dropout",Number(0));exact("resnet_out_scale_factor",Number(1))
      require(integer("addition_embed_type_num_heads",64)>0,"invalid inactive addition head count")
      take("sample_size",Null) match
        case Null => ()
        case Array(v) => require(v.size==2 && v.forall(x => integerValue(x,"sample_size")>0),"invalid sample size")
        case v => require(integerValue(v,"sample_size")>0,"invalid sample size")
      val channels=array("block_out_channels").map(integerValue(_,"block_out_channels"))
      require(channels.nonEmpty,"empty U-Net channels")
      def blocks(key: String,basic: String,cross: String): Vector[Boolean]=array(key).map {
        case Text(s) if s==basic => false
        case Text(s) if s==cross => true
        case _ => throw IllegalArgumentException(s"unsupported U-Net block: $key")
      }
      val down=blocks("down_block_types","DownBlock2D","CrossAttnDownBlock2D")
      val up=blocks("up_block_types","UpBlock2D","CrossAttnUpBlock2D")
      val time=take("time_embedding_dim",Null) match
        case Null => channels.head*4
        case v => integerValue(v,"time_embedding_dim")
      val config=UNetConfig(integer("in_channels",4),integer("out_channels",4),channels,down,up,
        integer("layers_per_block",2),integer("attention_head_dim",8),integer("cross_attention_dim",1280),
        integer("norm_num_groups",32),number("norm_eps",1e-5),integer("transformer_layers_per_block",1),
        integer("downsample_padding",1),time,boolean("flip_sin_to_cos",true),number("freq_shift",0),number("mid_block_scale_factor",1))
      require(fields.isEmpty,s"unknown U-Net config fields: ${fields.keys.toVector.sorted.mkString(", ")}")
      config
    }
    catch case e: IOException => throw IllegalArgumentException("invalid U-Net config JSON",e)
