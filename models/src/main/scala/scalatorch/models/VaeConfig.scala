package scalatorch.models

import scalatorch.nn.Module
import com.fasterxml.jackson.core.{JsonFactory,JsonToken,StreamReadConstraints,StreamReadFeature}
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import scala.collection.mutable
import scala.util.Using

/** AutoencoderKL structure and pipeline metadata; forward methods do not apply latent scaling. */
final case class VaeConfig(inChannels: Int,outChannels: Int,latentChannels: Int,channels: Vector[Int],
    layersPerBlock: Int,groups: Int,scalingFactor: Double=.18215,quantConv: Boolean=true,
    postQuantConv: Boolean=true,midAttention: Boolean=true,forceUpcast: Boolean=true,
    sampleHeight: Int=32,sampleWidth: Int=32):
  require(channels.nonEmpty && channels.size<=8 && (channels++Vector(inChannels,outChannels,latentChannels))
    .forall(n => n>0 && n<=16384),"invalid VAE channel/stage dimensions")
  require(layersPerBlock>0 && layersPerBlock<=16 && groups>0 && channels.forall(_%groups==0),"invalid VAE layers/groups")
  require(scalingFactor.isFinite && scalingFactor>0 && scalingFactor.toFloat>0 && scalingFactor.toFloat.isFinite &&
    (1/scalingFactor).toFloat.isFinite && (1/scalingFactor).toFloat>0,"invalid VAE scaling factor")
  require(sampleHeight>0 && sampleWidth>0 && sampleHeight<=16384 && sampleWidth<=16384,"invalid VAE sample size")

object VaeConfig:
  given Module[VaeConfig] with
    def namedParameters(value: VaeConfig) = Vector.empty
  private enum Value:
    case Number(n: BigDecimal)
    case Text(s: String)
    case Bool(b: Boolean)
    case Array(values: Vector[Value])
    case Null
  import Value.*
  private val factory: JsonFactory=JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(8).maxStringLength(1048576).build()).build()
  def read(path: Path): VaeConfig = Using.resource(Files.newInputStream(path)) { input =>
    val bytes=input.readNBytes(1048577)
    require(bytes.length<=1048576,"VAE config exceeds 1MiB")
    fromJson(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString)
  }
  def fromJson(json: String): VaeConfig =
    require(json.length<=1048576,"VAE config exceeds size limit")
    try Using.resource(factory.createParser(json)) { p =>
      def value(allowArray: Boolean=true): Value=p.currentToken() match
        case JsonToken.VALUE_NUMBER_INT | JsonToken.VALUE_NUMBER_FLOAT => Number(BigDecimal(p.getDecimalValue()))
        case JsonToken.VALUE_STRING => Text(p.getText())
        case JsonToken.VALUE_TRUE => Bool(true)
        case JsonToken.VALUE_FALSE => Bool(false)
        case JsonToken.VALUE_NULL => Null
        case JsonToken.START_ARRAY if allowArray =>
          val result=Vector.newBuilder[Value]
          var count=0
          while p.nextToken()!=JsonToken.END_ARRAY do
            require(count<32,"VAE config array too long")
            result+=value(false);count+=1
          Array(result.result())
        case _ => throw IllegalArgumentException("unsupported VAE config value")
      require(p.nextToken()==JsonToken.START_OBJECT,"VAE config must be an object")
      val fields=mutable.Map.empty[String,Value]
      while p.nextToken()!=JsonToken.END_OBJECT do
        require(p.currentToken()==JsonToken.FIELD_NAME && fields.size<128,"invalid VAE config field")
        val key=p.currentName();p.nextToken();fields(key)=value()
      require(p.nextToken()==null,"trailing VAE config JSON")
      def take(key: String,default: Value=Null): Value=fields.remove(key).getOrElse(default)
      def exact(key: String,expected: Value): Unit=require(take(key,expected)==expected,s"unsupported VAE option: $key")
      def intValue(v: Value,key: String): Int=v match
        case Number(n) if n.isWhole && n>=0 && n<=16384 => n.toInt
        case _ => throw IllegalArgumentException(s"invalid VAE integer: $key")
      def integer(key: String,default: Int): Int=intValue(take(key,Number(BigDecimal(default))),key)
      def boolean(key: String,default: Boolean): Boolean=take(key,Bool(default)) match
        case Bool(b) => b
        case _ => throw IllegalArgumentException(s"invalid VAE boolean: $key")
      def array(key: String): Vector[Value]=take(key) match
        case Array(values) => values
        case _ => throw IllegalArgumentException(s"missing VAE array: $key")
      exact("_class_name",Text("AutoencoderKL"));exact("act_fn",Text("silu"))
      exact("norm_type",Text("group"));exact("double_z",Bool(true))
      for key <- Vector("_diffusers_version","_name_or_path") do
        take(key) match
          case Null | Text(_) => ()
          case _ => throw IllegalArgumentException(s"invalid VAE metadata: $key")
      for key <- Vector("shift_factor","latents_mean","latents_std") do exact(key,Null)
      val channels=array("block_out_channels").map(intValue(_,"block_out_channels"))
      require(array("down_block_types")==Vector.fill(channels.size)(Text("DownEncoderBlock2D")),"unsupported VAE down blocks")
      require(array("up_block_types")==Vector.fill(channels.size)(Text("UpDecoderBlock2D")),"unsupported VAE up blocks")
      val (sampleHeight,sampleWidth)=take("sample_size") match
        case Null => (32,32)
        case Array(v) if v.size==2 => (intValue(v.head,"sample_size"),intValue(v(1),"sample_size"))
        case Array(_) => throw IllegalArgumentException("VAE sample_size must have two dimensions")
        case v => val size=intValue(v,"sample_size");(size,size)
      val scale=take("scaling_factor",Number(BigDecimal("0.18215"))) match
        case Number(n) => n.toDouble
        case _ => throw IllegalArgumentException("invalid VAE scaling factor")
      val c=VaeConfig(integer("in_channels",3),integer("out_channels",3),integer("latent_channels",4),channels,
        integer("layers_per_block",1),integer("norm_num_groups",32),scale,boolean("use_quant_conv",true),
        boolean("use_post_quant_conv",true),boolean("mid_block_add_attention",true),boolean("force_upcast",true),
        sampleHeight,sampleWidth)
      require(fields.isEmpty,s"unknown VAE config fields: ${fields.keys.toVector.sorted.mkString(", ")}")
      c
    }
    catch case e: IOException => throw IllegalArgumentException("invalid VAE config JSON",e)
