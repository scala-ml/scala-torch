package scalatorch.models

import scalatorch.nn.Module
import com.fasterxml.jackson.core.{JsonFactory, JsonToken, StreamReadConstraints, StreamReadFeature}
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import scala.collection.mutable
import scala.util.Using

final case class LlamaConfig(vocabSize: Int, hiddenSize: Int, intermediateSize: Int, numLayers: Int,
    numAttentionHeads: Int, numKeyValueHeads: Int, maxPositions: Int = 2048,
    rmsNormEps: Double = 1e-6, ropeTheta: Double = 10000.0, tieWordEmbeddings: Boolean = false,
    bosTokenId: Option[Long] = Some(1L), eosTokenIds: Vector[Long] = Vector(2L), initializerRange: Double = 0.02):
  require(vocabSize > 0 && hiddenSize > 0 && intermediateSize > 0 && numLayers > 0 && maxPositions > 0, "invalid Llama dimensions")
  require(numAttentionHeads > 0 && numKeyValueHeads > 0 && hiddenSize % numAttentionHeads == 0 &&
    numAttentionHeads % numKeyValueHeads == 0, "invalid Llama head dimensions")
  val headDim: Int = hiddenSize / numAttentionHeads
  require(headDim % 2 == 0, "Llama rotary head width must be even")
  require(rmsNormEps.isFinite && rmsNormEps > 0 && ropeTheta.isFinite && ropeTheta > 0, "invalid Llama norm/rotary configuration")
  require(initializerRange.isFinite && initializerRange >= 0, "invalid Llama initializer range")
  require((bosTokenId.toVector ++ eosTokenIds).forall(id => id >= 0 && id < vocabSize), "special token ID out of vocabulary")

object LlamaConfig:
  given Module[LlamaConfig] with
    def namedParameters(config: LlamaConfig) = Vector.empty
  private enum Value:
    case Integer(n: Long)
    case Number(n: Double)
    case Text(s: String)
    case Bool(b: Boolean)
    case Ids(ids: Vector[Long])
    case Null, Other
  private val factory: JsonFactory = JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(32).maxStringLength(1048576).build()).build()

  def read(path: Path): LlamaConfig = Using.resource(Files.newInputStream(path)) { input =>
    val bytes = input.readNBytes(1048577)
    require(bytes.length <= 1048576, "Llama config exceeds 1MiB")
    fromJson(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString)
  }

  def fromJson(json: String): LlamaConfig =
    require(json.length <= 1048576, "Llama config exceeds size limit")
    try Using.resource(factory.createParser(json)) { p =>
      require(p.nextToken() == JsonToken.START_OBJECT, "Llama config must be an object")
      val fields = mutable.Map.empty[String, Value]
      while p.nextToken() != JsonToken.END_OBJECT do
        require(p.currentToken() == JsonToken.FIELD_NAME, "invalid config field")
        val key = p.currentName()
        val token = p.nextToken()
        val value = token match
          case JsonToken.VALUE_NUMBER_INT => Value.Integer(p.getLongValue())
          case JsonToken.VALUE_NUMBER_FLOAT => Value.Number(p.getDoubleValue())
          case JsonToken.VALUE_STRING => Value.Text(p.getText())
          case JsonToken.VALUE_TRUE => Value.Bool(true)
          case JsonToken.VALUE_FALSE => Value.Bool(false)
          case JsonToken.VALUE_NULL => Value.Null
          case JsonToken.START_ARRAY if key == "eos_token_id" =>
            val ids = Vector.newBuilder[Long]
            var count = 0
            while p.nextToken() != JsonToken.END_ARRAY do
              require(p.currentToken() == JsonToken.VALUE_NUMBER_INT && count < 1024, "invalid EOS IDs")
              ids += p.getLongValue(); count += 1
            Value.Ids(ids.result())
          case JsonToken.START_ARRAY | JsonToken.START_OBJECT => p.skipChildren(); Value.Other
          case _ => throw IllegalArgumentException(s"invalid config value: $key")
        fields(key) = value
      require(p.nextToken() == null, "trailing config JSON")
      def integer(key: String, default: Option[Int] = None): Int = fields.get(key) match
        case Some(Value.Integer(n)) if n >= 0 && n <= Int.MaxValue => n.toInt
        case None if default.nonEmpty => default.get
        case _ => throw IllegalArgumentException(s"expected nonnegative integer config: $key")
      def number(key: String, default: Double): Double =
        val n = fields.get(key) match
          case Some(Value.Integer(n)) => n.toDouble
          case Some(Value.Number(n)) => n
          case None => default
          case _ => throw IllegalArgumentException(s"expected numeric config: $key")
        require(n.isFinite, s"nonfinite config: $key"); n
      def bool(key: String, default: Boolean): Boolean = fields.get(key) match
        case Some(Value.Bool(b)) => b
        case None => default
        case _ => throw IllegalArgumentException(s"expected boolean config: $key")
      require(fields.get("model_type").contains(Value.Text("llama")), "only model_type=llama is supported")
      require(fields.getOrElse("hidden_act", Value.Text("silu")) == Value.Text("silu"), "only SiLU Llama MLP is supported")
      for key <- Seq("rope_scaling", "rope_parameters", "sliding_window") do
        require(!fields.contains(key) || fields(key) == Value.Null, s"unsupported Llama option: $key")
      require(!bool("attention_bias", false) && !bool("mlp_bias", false), "Llama projection biases unsupported")
      require(number("attention_dropout", 0.0) == 0 && integer("pretraining_tp", Some(1)) == 1, "unsupported Llama dropout/tensor parallel setting")
      require(number("partial_rotary_factor", 1.0) == 1.0, "partial rotary dimensions unsupported")
      val heads = integer("num_attention_heads")
      val bos = fields.get("bos_token_id") match
        case None => Some(1L)
        case Some(Value.Null) => None
        case Some(Value.Integer(n)) => Some(n)
        case _ => throw IllegalArgumentException("invalid BOS ID")
      val eos = fields.get("eos_token_id") match
        case None => Vector(2L)
        case Some(Value.Null) => Vector.empty
        case Some(Value.Integer(n)) => Vector(n)
        case Some(Value.Ids(ids)) => ids
        case _ => throw IllegalArgumentException("invalid EOS IDs")
      val config = LlamaConfig(integer("vocab_size"), integer("hidden_size"), integer("intermediate_size"),
        integer("num_hidden_layers"), heads, integer("num_key_value_heads", Some(heads)), integer("max_position_embeddings", Some(2048)),
        number("rms_norm_eps", 1e-6), number("rope_theta", 10000.0), bool("tie_word_embeddings", false), bos, eos,
        number("initializer_range", 0.02))
      require(integer("head_dim", Some(config.headDim)) == config.headDim, "custom Llama head_dim unsupported")
      config
    }
    catch case e: IOException => throw IllegalArgumentException("invalid Llama config JSON", e)
