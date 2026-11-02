package scalatorch.io

import scalatorch.Shape
import com.fasterxml.jackson.core.{JsonFactory, JsonParser, JsonToken, StreamReadConstraints, StreamReadFeature}
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import scala.collection.mutable
import scala.util.Using

private[io] final case class Header(tensors: Map[String, TensorInfo], metadata: Map[String, String])

private[io] object SafeTensorsHeader:
  private val factory: JsonFactory = JsonFactory.builder()
    .enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(8).maxNumberLength(20).maxStringLength(1000000).build())
    .build()

  private def text(value: String): String =
    var i = 0
    while i < value.length do
      val c = value.charAt(i)
      if Character.isHighSurrogate(c) then
        require(i + 1 < value.length && Character.isLowSurrogate(value.charAt(i + 1)), "unpaired JSON surrogate")
        i += 1
      else require(!Character.isLowSurrogate(c), "unpaired JSON surrogate")
      i += 1
    value

  private def string(parser: JsonParser): String =
    require(parser.currentToken() == JsonToken.VALUE_STRING, "expected JSON string")
    text(parser.getText())

  private def integers(parser: JsonParser, maximum: Int): Vector[Long] =
    require(parser.currentToken() == JsonToken.START_ARRAY, "expected integer array")
    val out = Vector.newBuilder[Long]
    var size = 0
    while parser.nextToken() != JsonToken.END_ARRAY do
      require(parser.currentToken() == JsonToken.VALUE_NUMBER_INT && size < maximum, "invalid integer array or rank limit")
      val n = parser.getLongValue()
      require(n >= 0, "negative shape or offset")
      out += n
      size += 1
    out.result()

  private def tensor(parser: JsonParser, limits: ReadLimits): TensorInfo =
    require(parser.currentToken() == JsonToken.START_OBJECT, "expected tensor object")
    var dtype: Option[SafeDType] = None
    var shape: Option[Shape] = None
    var offsets: Option[Vector[Long]] = None
    while parser.nextToken() != JsonToken.END_OBJECT do
      require(parser.currentToken() == JsonToken.FIELD_NAME, "invalid tensor field")
      val key = text(parser.currentName())
      parser.nextToken()
      key match
        case "dtype" =>
          val name = string(parser)
          dtype = Some(SafeDType.values.find(_.toString == name).getOrElse(throw IllegalArgumentException(s"unsupported safetensors dtype: $name")))
        case "shape" => shape = Some(Shape.from(integers(parser, limits.maxRank)))
        case "data_offsets" =>
          val values = integers(parser, 2)
          require(values.size == 2, "data_offsets must contain two integers")
          offsets = Some(values)
        case _ => throw IllegalArgumentException(s"unknown tensor field: $key")
    require(dtype.nonEmpty && shape.nonEmpty && offsets.nonEmpty, "missing tensor metadata")
    val ds = offsets.get
    val info = TensorInfo(dtype.get, shape.get, ds(0), ds(1))
    require(info.end >= info.begin, "reversed tensor offsets")
    require(Math.multiplyExact(info.shape.numel, info.dtype.bytes.toLong) == info.end - info.begin, "tensor byte count mismatch")
    info

  def parse(bytes: Array[Byte], dataLength: Long, limits: ReadLimits): Header =
    require(bytes.nonEmpty && bytes(0) == '{'.toByte, "safetensors header must begin with object")
    val json = UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString
    Using.resource(factory.createParser(json)) { parser =>
      require(parser.nextToken() == JsonToken.START_OBJECT, "expected header object")
      val tensors = mutable.Map.empty[String, TensorInfo]
      val metadata = mutable.Map.empty[String, String]
      while parser.nextToken() != JsonToken.END_OBJECT do
        require(parser.currentToken() == JsonToken.FIELD_NAME, "invalid header field")
        val name = text(parser.currentName())
        parser.nextToken()
        if name == "__metadata__" then
          require(parser.currentToken() == JsonToken.START_OBJECT, "metadata must be string map")
          while parser.nextToken() != JsonToken.END_OBJECT do
            require(parser.currentToken() == JsonToken.FIELD_NAME, "invalid metadata key")
            val key = text(parser.currentName())
            parser.nextToken()
            metadata(key) = string(parser)
        else
          require(tensors.size < limits.maxTensors, "tensor count limit exceeded")
          tensors(name) = tensor(parser, limits)
      require(parser.nextToken() == null, "trailing header JSON")
      var cursor = 0L
      tensors.values.toVector.sortBy(info => (info.begin, info.end)).foreach { info =>
        require(info.begin == cursor && info.end <= dataLength, "tensor offsets overlap, contain gaps, or exceed file")
        cursor = info.end
      }
      require(cursor == dataLength, "unindexed trailing data")
      Header(tensors.toMap, metadata.toMap)
    }
