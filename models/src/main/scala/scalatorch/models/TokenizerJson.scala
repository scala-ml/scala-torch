package scalatorch.models

import com.fasterxml.jackson.core.{JsonFactory, JsonToken, StreamReadConstraints, StreamReadFeature}
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import scala.util.Using

private[models] object TokenizerJson:
  enum Value:
    case Obj(fields: Map[String, Value])
    case Arr(values: Vector[Value])
    case Str(value: String)
    case Num(value: Long)
    case Bool(value: Boolean)
    case Null
    def obj: Map[String, Value] = this match
      case Obj(fields) => fields
      case _ => throw IllegalArgumentException("expected tokenizer object")
    def arr: Vector[Value] = this match
      case Arr(values) => values
      case _ => throw IllegalArgumentException("expected tokenizer array")
    def str: String = this match
      case Str(value) => value
      case _ => throw IllegalArgumentException("expected tokenizer string")
    def int: Int = this match
      case Num(value) if value >= 0 && value <= Int.MaxValue => value.toInt
      case _ => throw IllegalArgumentException("expected nonnegative tokenizer integer")
  val maxBytes = 32 * 1024 * 1024
  private val factory: JsonFactory = JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(16).maxStringLength(maxBytes).build()).build()

  def read(path: Path): Value = Using.resource(Files.newInputStream(path)) { input =>
    val bytes = input.readNBytes(maxBytes + 1)
    require(bytes.length <= maxBytes, "tokenizer exceeds 32MiB")
    parse(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString)
  }

  def validUnicode(s: String): Boolean =
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      if Character.isHighSurrogate(c) then
        if i + 1 == s.length || !Character.isLowSurrogate(s.charAt(i + 1)) then return false
        i += 2
      else if Character.isLowSurrogate(c) then return false
      else i += 1
    true

  def parse(json: String): Value =
    require(json.length <= maxBytes && validUnicode(json), "invalid or oversized tokenizer JSON")
    try Using.resource(factory.createParser(json)) { p =>
      def value(): Value = p.currentToken() match
        case JsonToken.START_OBJECT =>
          val fields = Map.newBuilder[String, Value]
          while p.nextToken() != JsonToken.END_OBJECT do
            require(p.currentToken() == JsonToken.FIELD_NAME, "invalid tokenizer field")
            val name = p.currentName()
            require(validUnicode(name), "invalid Unicode tokenizer key")
            p.nextToken(); fields += name -> value()
          Value.Obj(fields.result())
        case JsonToken.START_ARRAY =>
          val values = Vector.newBuilder[Value]
          while p.nextToken() != JsonToken.END_ARRAY do values += value()
          Value.Arr(values.result())
        case JsonToken.VALUE_STRING =>
          val text = p.getText()
          require(validUnicode(text), "invalid Unicode tokenizer value")
          Value.Str(text)
        case JsonToken.VALUE_NUMBER_INT => Value.Num(p.getLongValue())
        case JsonToken.VALUE_TRUE => Value.Bool(true)
        case JsonToken.VALUE_FALSE => Value.Bool(false)
        case JsonToken.VALUE_NULL => Value.Null
        case _ => throw IllegalArgumentException("unsupported tokenizer JSON value")
      p.nextToken()
      val result = value()
      require(p.nextToken() == null, "trailing tokenizer JSON")
      result
    }
    catch case e: IOException => throw IllegalArgumentException("invalid tokenizer JSON", e)
