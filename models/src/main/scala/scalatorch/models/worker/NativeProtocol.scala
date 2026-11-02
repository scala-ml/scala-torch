package scalatorch.models.worker

import com.fasterxml.jackson.core.{JsonFactory,JsonToken,StreamReadConstraints,StreamReadFeature}
import java.io.{IOException,OutputStream,StringWriter}
import java.nio.charset.StandardCharsets.UTF_8
import scala.util.Using

enum NativeJson:
  case Obj(fields: Map[String,NativeJson])
  case Arr(values: Vector[NativeJson])
  case Str(value: String)
  case Number(value: BigDecimal)
  case Bool(value: Boolean)
  case Null

object NativeJson:
  val maxChars=16*1024*1024
  private val factory: JsonFactory=JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(32).maxStringLength(maxChars).build()).build()
  def parse(text: String): NativeJson =
    require(text.length<=maxChars && scalatorch.models.TokenizerJson.validUnicode(text),"invalid or oversized worker JSON")
    try Using.resource(factory.createParser(text)) { p =>
      def value(): NativeJson=p.currentToken() match
        case JsonToken.START_OBJECT =>
          val fields=Map.newBuilder[String,NativeJson]
          while p.nextToken()!=JsonToken.END_OBJECT do
            require(p.currentToken()==JsonToken.FIELD_NAME,"invalid JSON field")
            val name=p.currentName()
            require(scalatorch.models.TokenizerJson.validUnicode(name),"invalid Unicode field")
            p.nextToken();fields+=name->value()
          Obj(fields.result())
        case JsonToken.START_ARRAY =>
          val values=Vector.newBuilder[NativeJson]
          while p.nextToken()!=JsonToken.END_ARRAY do values+=value()
          Arr(values.result())
        case JsonToken.VALUE_STRING =>
          val s=p.getText()
          require(scalatorch.models.TokenizerJson.validUnicode(s),"invalid Unicode string")
          Str(s)
        case JsonToken.VALUE_NUMBER_INT | JsonToken.VALUE_NUMBER_FLOAT => Number(BigDecimal(p.getDecimalValue()))
        case JsonToken.VALUE_TRUE => Bool(true)
        case JsonToken.VALUE_FALSE => Bool(false)
        case JsonToken.VALUE_NULL => Null
        case _ => throw IllegalArgumentException("invalid worker JSON value")
      p.nextToken()
      val result=value()
      require(p.nextToken()==null,"trailing worker JSON")
      result
    }
    catch case e: IOException => throw IllegalArgumentException("invalid worker JSON",e)
  def render(value: NativeJson): String =
    val out=new StringWriter()
    Using.resource(factory.createGenerator(out)) { g =>
      def write(v: NativeJson): Unit=v match
        case Obj(fields) =>
          g.writeStartObject()
          fields.toVector.sortBy(_._1).foreach { (key,entry) => g.writeFieldName(key);write(entry) }
          g.writeEndObject()
        case Arr(values) => g.writeStartArray();values.foreach(write);g.writeEndArray()
        case Str(s) => g.writeString(s)
        case Number(n) => g.writeNumber(n.bigDecimal)
        case Bool(b) => g.writeBoolean(b)
        case Null => g.writeNull()
      write(value)
    }
    out.toString

final case class NativeRequest(id: String,kind: String,params: Map[String,NativeJson],owner: Option[String])
object NativeRequest:
  import NativeJson.*
  def parse(line: String): NativeRequest =
    val fields=NativeJson.parse(line) match
      case Obj(values) => values
      case _ => throw IllegalArgumentException("request must be an object")
    require(fields.keySet.subsetOf(Set("id","type","params","owner_id")),"unsupported envelope fields")
    def string(key: String): String=fields.get(key) match
      case Some(Str(s)) if s.nonEmpty && s.length<=256 => s
      case _ => throw IllegalArgumentException(s"invalid request $key")
    val params=fields.getOrElse("params",Null) match
      case Null => Map.empty[String,NativeJson]
      case Obj(values) => values
      case _ => throw IllegalArgumentException("params must be an object")
    NativeRequest(string("id"),string("type"),params,if fields.contains("owner_id") then Some(string("owner_id")) else None)

/** Single writer lock and explicit UTF-8; does not close the caller-owned stream. */
final class NativeEmitter(out: OutputStream):
  import NativeJson.*
  def emit(fields: Map[String,NativeJson]): Unit=synchronized {
    val line=NativeJson.render(Obj(fields))+"\n"
    out.write(line.getBytes(UTF_8));out.flush()
  }
  def result(id: String,fields: Map[String,NativeJson]): Unit=
    emit(Map("id"->Str(id),"type"->Str("result"),"result"->Obj(fields)))
  def hello(features: Vector[String]): Unit=
    emit(Map("id"->Null,"type"->Str("hello"),"protocol"->Number(BigDecimal(1)),
      "engine_version"->Str("scala-torch-0.1.0"),"python"->Str(""),"cuda"->Bool(false),
      "device"->Str("cpu"),"dry_run"->Bool(false),"features"->Arr(features.map(Str.apply))))
  def event(id: String,name: String,fields: Map[String,NativeJson]): Unit=
    require(fields.keySet.intersect(Set("id","type","event")).isEmpty,"reserved event field")
    emit(fields++Map("id"->Str(id),"type"->Str("event"),"event"->Str(name)))
  def error(id: Option[String],code: String,message: String,detail: String=""): Unit=
    emit(Map("id"->id.map(Str.apply).getOrElse(Null),"type"->Str("error"),"code"->Str(code),
      "message"->Str(message),"detail"->Str(detail)))
