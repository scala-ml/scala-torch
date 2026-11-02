package scalatorch.io

import com.fasterxml.jackson.core.{JsonFactory, JsonToken, StreamReadConstraints, StreamReadFeature}
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import scala.collection.mutable
import scala.util.Using

final case class ShardLimits(maxIndexBytes: Int = 16777216, maxShards: Int = 128):
  require(maxIndexBytes >= 2 && maxIndexBytes < Int.MaxValue && maxShards > 0, "invalid shard limits")

private[io] final case class ShardIndex(weights: Map[String,String], totalBytes: Option[Long])

private[io] object SafeTensorsIndex:
  private val factory: JsonFactory = JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(16).maxStringLength(1000000).build()).build()
  private def text(s: String): String =
    var i = 0
    while i < s.length do
      if Character.isHighSurrogate(s.charAt(i)) then
        require(i+1 < s.length && Character.isLowSurrogate(s.charAt(i+1)), "unpaired index surrogate")
        i += 1
      else require(!Character.isLowSurrogate(s.charAt(i)), "unpaired index surrogate")
      i += 1
    s

  def read(path: Path, limits: ShardLimits, maxTensors: Int): ShardIndex =
    try Using.resource(Files.newInputStream(path)) { input =>
      val bytes = input.readNBytes(limits.maxIndexBytes+1)
      require(bytes.length <= limits.maxIndexBytes, "safetensors index exceeds size limit")
      val json = UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString
      Using.resource(factory.createParser(json)) { p =>
        require(p.nextToken() == JsonToken.START_OBJECT, "index must be an object")
        val weights = mutable.Map.empty[String,String]
        var foundMap = false
        var total: Option[Long] = None
        while p.nextToken() != JsonToken.END_OBJECT do
          require(p.currentToken() == JsonToken.FIELD_NAME, "invalid index field")
          val field = p.currentName()
          p.nextToken()
          field match
            case "weight_map" =>
              foundMap = true
              require(p.currentToken() == JsonToken.START_OBJECT, "weight_map must be an object")
              while p.nextToken() != JsonToken.END_OBJECT do
                require(p.currentToken() == JsonToken.FIELD_NAME && weights.size < maxTensors, "invalid or oversized weight_map")
                val name = text(p.currentName())
                require(name.nonEmpty, "empty tensor name in index")
                require(p.nextToken() == JsonToken.VALUE_STRING, "shard name must be a string")
                val file = text(p.getText())
                require(file.endsWith(".safetensors") && !file.exists(c => c == '/' || c == '\\' || c == ':' || c == 0), "shard must be a simple safetensors basename")
                weights(name) = file
            case "metadata" =>
              require(p.currentToken() == JsonToken.START_OBJECT, "index metadata must be an object")
              while p.nextToken() != JsonToken.END_OBJECT do
                require(p.currentToken() == JsonToken.FIELD_NAME, "invalid index metadata")
                val key = p.currentName()
                p.nextToken()
                if key == "total_size" then
                  require(p.currentToken() == JsonToken.VALUE_NUMBER_INT, "total_size must be an integer")
                  val n = p.getLongValue()
                  require(n >= 0, "negative index total_size")
                  total = Some(n)
                else p.skipChildren()
            case _ => throw IllegalArgumentException(s"unsupported index field: $field")
        require(foundMap && weights.nonEmpty && p.nextToken() == null, "empty or trailing safetensors index")
        require(weights.values.toSet.size <= limits.maxShards, "shard count exceeds limit")
        ShardIndex(weights.toMap,total)
      }
    }
    catch case e: IOException => throw IllegalArgumentException("invalid safetensors index",e)
