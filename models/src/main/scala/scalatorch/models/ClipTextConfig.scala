package scalatorch.models

import com.fasterxml.jackson.core.{JsonFactory,JsonToken,StreamReadConstraints,StreamReadFeature}
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import scala.collection.mutable
import scala.util.Using

final case class ClipTextConfig(vocabulary: Int,width: Int,intermediate: Int,layers: Int,heads: Int,
    maxPositions: Int,epsilon: Double=1e-5,activation: ClipActivation=ClipActivation.QuickGelu,
    bosId: Int=0,eosId: Int=2,padId: Int=1):
  require(vocabulary>0 && vocabulary<=2000000 && width>0 && width<=16384 && intermediate>0 && intermediate<=65536,
    "invalid CLIP dimensions")
  require(layers>0 && layers<=512 && heads>0 && width%heads==0 && maxPositions>0 && maxPositions<=65536,
    "invalid CLIP depth/heads/positions")
  require(epsilon.isFinite && epsilon>0,"invalid CLIP normalization epsilon")
  require(Vector(bosId,eosId,padId).forall(id => id>=0 && id<vocabulary),"CLIP special token outside vocabulary")

object ClipTextConfig:
  private enum Value:
    case Number(n: BigDecimal)
    case Text(s: String)
    case Bool(b: Boolean)
    case Strings(v: Vector[String])
    case Null
  import Value.*
  private val factory: JsonFactory=JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(8).maxStringLength(1048576).build()).build()
  def read(path: Path): ClipTextConfig = Using.resource(Files.newInputStream(path)) { input =>
    val bytes=input.readNBytes(1048577)
    require(bytes.length<=1048576,"CLIP config exceeds 1MiB")
    fromJson(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString)
  }
  def fromJson(json: String): ClipTextConfig =
    require(json.length<=1048576,"CLIP config exceeds size limit")
    try Using.resource(factory.createParser(json)) { p =>
      require(p.nextToken()==JsonToken.START_OBJECT,"CLIP config must be an object")
      val fields=mutable.Map.empty[String,Value]
      while p.nextToken()!=JsonToken.END_OBJECT do
        require(p.currentToken()==JsonToken.FIELD_NAME && fields.size<128,"invalid CLIP config field")
        val key=p.currentName()
        val value=p.nextToken() match
          case JsonToken.VALUE_NUMBER_INT | JsonToken.VALUE_NUMBER_FLOAT => Number(BigDecimal(p.getDecimalValue()))
          case JsonToken.VALUE_STRING => Text(p.getText())
          case JsonToken.VALUE_TRUE => Bool(true)
          case JsonToken.VALUE_FALSE => Bool(false)
          case JsonToken.VALUE_NULL => Null
          case JsonToken.START_ARRAY =>
            val values=Vector.newBuilder[String]
            var count=0
            while p.nextToken()!=JsonToken.END_ARRAY do
              require(p.currentToken()==JsonToken.VALUE_STRING && count<32,"invalid CLIP string array")
              values+=p.getText();count+=1
            Strings(values.result())
          case _ => throw IllegalArgumentException(s"unsupported CLIP config value: $key")
        fields(key)=value
      require(p.nextToken()==null,"trailing CLIP JSON")
      def take(key: String,default: Value=Null): Value=fields.remove(key).getOrElse(default)
      def exact(key: String,expected: Value): Unit=require(take(key,expected)==expected,s"unsupported CLIP option: $key")
      def integer(key: String,default: Value=Null): Int=take(key,default) match
        case Number(n) if n.isWhole && n>=0 && n<=Int.MaxValue => n.toInt
        case _ => throw IllegalArgumentException(s"invalid CLIP integer: $key")
      def number(key: String,default: Double): Double=take(key,Number(BigDecimal(default))) match
        case Number(n) if n.toDouble.isFinite => n.toDouble
        case _ => throw IllegalArgumentException(s"invalid CLIP number: $key")
      require(take("model_type")==Text("clip_text_model"),"only CLIP text model supported")
      exact("architectures",Strings(Vector("CLIPTextModel")))
      exact("attention_dropout",Number(0));exact("dropout",Number(0));exact("use_attention_mask",Bool(false))
      exact("is_decoder",Bool(false));exact("add_cross_attention",Bool(false))
      for key <- Vector("_name_or_path","transformers_version") do
        take(key) match
          case Null | Text(_) => ()
          case _ => throw IllegalArgumentException(s"invalid CLIP metadata: $key")
      require(Set[Value](Null,Text("float32"),Text("float16"),Text("bfloat16"))(take("torch_dtype")),"unsupported CLIP dtype metadata")
      require(number("initializer_factor",1)>0 && number("initializer_range",.02)>=0,"invalid CLIP initializer metadata")
      require(integer("projection_dim",Number(512))>0,"invalid CLIP projection metadata")
      val activation=take("hidden_act",Text("quick_gelu")) match
        case Text("quick_gelu") => ClipActivation.QuickGelu
        case Text("gelu") => ClipActivation.Gelu
        case Text("gelu_new") | Text("gelu_pytorch_tanh") => ClipActivation.GeluTanh
        case _ => throw IllegalArgumentException("unsupported CLIP activation")
      val c=ClipTextConfig(integer("vocab_size"),integer("hidden_size"),integer("intermediate_size"),
        integer("num_hidden_layers"),integer("num_attention_heads"),integer("max_position_embeddings"),
        number("layer_norm_eps",1e-5),activation,integer("bos_token_id",Number(0)),integer("eos_token_id",Number(2)),
        integer("pad_token_id",Number(1)))
      require(fields.isEmpty,s"unknown CLIP config fields: ${fields.keys.toVector.sorted.mkString(", ")}")
      c
    }
    catch case e: IOException => throw IllegalArgumentException("invalid CLIP config JSON",e)
