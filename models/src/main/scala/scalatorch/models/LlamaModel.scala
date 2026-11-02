package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd, Variable}
import scalatorch.nn.*
import scalatorch.io.{ReadLimits, SafeDType, SafeTensorSet, ShardLimits}
import java.nio.file.{Files, Path}
import scala.util.Using

final case class LlamaLimits(maxLayers: Int = 512, maxParameters: Long = 1000000000L,
    maxLogitsElements: Long = 16777216L, maxScoreElements: Long = 16777216L):
  require(maxLayers >= 0 && maxParameters >= 0 && maxLogitsElements >= 0 && maxScoreElements >= 0, "invalid Llama limits")
object LlamaLimits:
  given Module[LlamaLimits] with
    def namedParameters(limits: LlamaLimits) = Vector.empty

final case class LlamaAttention(qProj: Linear, kProj: Linear, vProj: Linear, oProj: Linear) derives Module:
  def apply(input: Variable, config: LlamaConfig, maxScores: Long): Variable =
    run(input,config,maxScores,0,None)
  private[models] def cached(input: Variable, config: LlamaConfig, maxScores: Long,
      position: Int, cache: LlamaCache.Layer): Variable = run(input,config,maxScores,position,Some(cache))
  private def run(input: Variable, config: LlamaConfig, maxScores: Long,
      position: Int, cache: Option[LlamaCache.Layer]): Variable =
    val b = input.data.shape.dims(0); val t = input.data.shape.dims(1)
    val flat = input.reshape(Shape.from(Seq(b * t, config.hiddenSize.toLong)))
    def heads(layer: Linear, count: Int): Variable = layer(flat)
      .reshape(Shape.from(Seq(b,t,count.toLong,config.headDim.toLong))).transpose(1,2)
    val q = heads(qProj, config.numAttentionHeads).rope(positionStart=position,theta=config.ropeTheta)
    val freshKey = heads(kProj, config.numKeyValueHeads).rope(positionStart=position,theta=config.ropeTheta)
    val freshValue = heads(vProj, config.numKeyValueHeads)
    val (k,v) = cache.fold((freshKey,freshValue)) { layer =>
      val (key,value) = layer.append(freshKey.data,freshValue.data,position)
      (Variable(key,false),Variable(value,false))
    }
    val attended = q.causalAttention(k, v, queryStart=position,maxScoreElements=maxScores).transpose(1,2)
      .reshape(Shape.from(Seq(b*t, config.hiddenSize.toLong)))
    oProj(attended).reshape(input.data.shape)

final case class LlamaMLP(gateProj: Linear, upProj: Linear, downProj: Linear) derives Module:
  def apply(input: Variable): Variable =
    val width = input.data.shape.dims.last
    val flat = input.reshape(Shape.from(Seq(input.data.shape.numel / width, width)))
    downProj(gateProj(flat).silu() * upProj(flat)).reshape(input.data.shape)

final case class LlamaBlock(inputNorm: RMSNorm, attention: LlamaAttention, postNorm: RMSNorm, mlp: LlamaMLP) derives Module:
  def apply(input: Variable, config: LlamaConfig, maxScores: Long): Variable =
    val h = input + attention(inputNorm(input), config, maxScores)
    h + mlp(postNorm(h))
  private[models] def cached(input: Variable, config: LlamaConfig, maxScores: Long,
      position: Int, cache: LlamaCache.Layer): Variable =
    val h = input + attention.cached(inputNorm(input),config,maxScores,position,cache)
    h + mlp(postNorm(h))

final case class LlamaModel(config: LlamaConfig, embedding: Embedding, layers: Vector[LlamaBlock], norm: RMSNorm,
    lmHead: Linear, limits: LlamaLimits) derives Module:
  def forward(tokens: Tensor[Long]): Variable =
    require(tokens.shape.rank == 2, "Llama tokens must have shape (batch,tokens)")
    val b = tokens.shape.dims(0); val t = tokens.shape.dims(1)
    require(t <= config.maxPositions, "Llama context limit exceeded")
    val shape = Shape.from(Seq(b,t,config.vocabSize.toLong))
    require(shape.numel <= limits.maxLogitsElements && shape.numel <= Int.MaxValue, "Llama logits allocation limit exceeded")
    val hidden = layers.foldLeft(embedding(tokens))((x, layer) => layer(x, config, limits.maxScoreElements))
    val normalized = norm(hidden).reshape(Shape.from(Seq(b*t, config.hiddenSize.toLong)))
    lmHead(normalized).reshape(shape)
  def logits(tokens: Tensor[Long]): Tensor[Float] = Autograd.noGrad { forward(tokens).data }
  def newCache(batchSize: Int = 1, capacity: Int = config.maxPositions, maxBytes: Long = 268435456L): LlamaCache =
    new LlamaCache(this,batchSize,capacity,maxBytes)
  def generateIds(prompt: Seq[Long], maxNewTokens: Int, useCache: Boolean = true,
      sampling: SamplingConfig = SamplingConfig()): Vector[Long] =
    require(prompt.nonEmpty && maxNewTokens >= 0 && prompt.length.toLong + maxNewTokens <= config.maxPositions, "invalid generation length")
    require(prompt.forall(id => id >= 0 && id < config.vocabSize), "generation token outside vocabulary")
    var ids = prompt.toVector
    if maxNewTokens == 0 then return ids
    val sampler = new TokenSampler(sampling)
    var generated = 0
    var stopped = false
    val cache = if useCache then Some(newCache(capacity=prompt.length+maxNewTokens)) else None
    try
      var chunk = ids.toArray
      while generated < maxNewTokens && !stopped do
        val input = Tensor.fromLongs(chunk, Shape.from(Seq(1L,chunk.length.toLong)))
        val result = cache.fold(logits(input))(_.decode(input))
        val scores = result.slice(1,chunk.length-1L,chunk.length.toLong).toFloatArray
        val next = sampler.sample(scores)
        ids = ids :+ next
        generated += 1
        stopped = config.eosTokenIds.contains(next)
        chunk = if useCache then Array(next) else ids.toArray
    finally cache.foreach(_.close())
    ids
  def namedWeights: Vector[(String, Parameter)] =
    val base = Vector("model.embed_tokens.weight" -> embedding.weight, "model.norm.weight" -> norm.weight, "lm_head.weight" -> lmHead.weight)
    base ++ layers.zipWithIndex.flatMap { (layer, i) =>
      val prefix = s"model.layers.$i"
      Vector(s"$prefix.input_layernorm.weight" -> layer.inputNorm.weight,
        s"$prefix.post_attention_layernorm.weight" -> layer.postNorm.weight,
        s"$prefix.self_attn.q_proj.weight" -> layer.attention.qProj.weight,
        s"$prefix.self_attn.k_proj.weight" -> layer.attention.kProj.weight,
        s"$prefix.self_attn.v_proj.weight" -> layer.attention.vProj.weight,
        s"$prefix.self_attn.o_proj.weight" -> layer.attention.oProj.weight,
        s"$prefix.mlp.gate_proj.weight" -> layer.mlp.gateProj.weight,
        s"$prefix.mlp.up_proj.weight" -> layer.mlp.upProj.weight,
        s"$prefix.mlp.down_proj.weight" -> layer.mlp.downProj.weight)
    }

object LlamaModel:
  private def validate(config: LlamaConfig, limits: LlamaLimits): Unit =
    require(config.numLayers <= limits.maxLayers, "Llama layer limit exceeded")
    val h = BigInt(config.hiddenSize)
    val perLayer = 2*h*h + 2*BigInt(config.numKeyValueHeads)*config.headDim*h +
      3*h*config.intermediateSize + 2*h
    val total = h*config.vocabSize*(if config.tieWordEmbeddings then 1 else 2) + perLayer*config.numLayers + h
    require(total <= limits.maxParameters, "Llama parameter limit exceeded")

  private def shapes(c: LlamaConfig): Vector[(String, Shape)] =
    def s(ds: Int*) = Shape.from(ds.map(_.toLong))
    Vector("model.embed_tokens.weight" -> s(c.vocabSize,c.hiddenSize), "model.norm.weight" -> s(c.hiddenSize),
      "lm_head.weight" -> s(c.vocabSize,c.hiddenSize)) ++ (0 until c.numLayers).toVector.flatMap { i =>
      val p = s"model.layers.$i"
      Vector(s"$p.input_layernorm.weight" -> s(c.hiddenSize), s"$p.post_attention_layernorm.weight" -> s(c.hiddenSize),
        s"$p.self_attn.q_proj.weight" -> s(c.hiddenSize,c.hiddenSize),
        s"$p.self_attn.k_proj.weight" -> s(c.numKeyValueHeads*c.headDim,c.hiddenSize),
        s"$p.self_attn.v_proj.weight" -> s(c.numKeyValueHeads*c.headDim,c.hiddenSize),
        s"$p.self_attn.o_proj.weight" -> s(c.hiddenSize,c.hiddenSize),
        s"$p.mlp.gate_proj.weight" -> s(c.intermediateSize,c.hiddenSize),
        s"$p.mlp.up_proj.weight" -> s(c.intermediateSize,c.hiddenSize),
        s"$p.mlp.down_proj.weight" -> s(c.hiddenSize,c.intermediateSize))
    }

  private def build(c: LlamaConfig, trainable: Boolean, limits: LlamaLimits)(read: (String, Shape) => Tensor[Float]): LlamaModel =
    def parameter(name: String, ds: Int*) = Parameter(read(name, Shape.from(ds.map(_.toLong))), trainable)
    def linear(name: String, in: Int, out: Int) = Linear(parameter(s"$name.weight", out, in), None)
    def norm(name: String) = RMSNorm(parameter(s"$name.weight", c.hiddenSize), c.rmsNormEps)
    val embedding = Embedding(parameter("model.embed_tokens.weight", c.vocabSize,c.hiddenSize))
    val layers = Vector.tabulate(c.numLayers) { i =>
      val p = s"model.layers.$i"
      LlamaBlock(norm(s"$p.input_layernorm"), LlamaAttention(
        linear(s"$p.self_attn.q_proj", c.hiddenSize,c.hiddenSize),
        linear(s"$p.self_attn.k_proj", c.hiddenSize,c.numKeyValueHeads*c.headDim),
        linear(s"$p.self_attn.v_proj", c.hiddenSize,c.numKeyValueHeads*c.headDim),
        linear(s"$p.self_attn.o_proj", c.hiddenSize,c.hiddenSize)), norm(s"$p.post_attention_layernorm"),
        LlamaMLP(linear(s"$p.mlp.gate_proj", c.hiddenSize,c.intermediateSize),
          linear(s"$p.mlp.up_proj", c.hiddenSize,c.intermediateSize), linear(s"$p.mlp.down_proj", c.intermediateSize,c.hiddenSize)))
    }
    val head = if c.tieWordEmbeddings then Linear(embedding.weight, None) else linear("lm_head", c.hiddenSize,c.vocabSize)
    LlamaModel(c, embedding, layers, norm("model.norm"), head, limits)

  def init(config: LlamaConfig, seed: Long, trainable: Boolean = true, limits: LlamaLimits = LlamaLimits()): LlamaModel =
    validate(config, limits)
    val random = new java.util.Random(seed)
    build(config, trainable, limits) { (name, shape) =>
      require(shape.numel <= Int.MaxValue, "Llama weight exceeds Int indexing")
      val values = if name.contains("layernorm") || name == "model.norm.weight" then Array.fill(shape.numel.toInt)(1f)
        else Array.fill(shape.numel.toInt)((random.nextGaussian() * config.initializerRange).toFloat)
      Tensor.fromFloats(values, shape)
    }

  def load(directory: Path, trainable: Boolean = false, limits: LlamaLimits = LlamaLimits(),
      readLimits: ReadLimits = ReadLimits(), shardLimits: ShardLimits = ShardLimits()): LlamaModel =
    val c = LlamaConfig.read(directory.resolve("config.json"))
    validate(c, limits)
    Using.resource(SafeTensorSet.open(directory, readLimits, shardLimits)) { archive =>
      val expected = shapes(c).toMap
      val optional = if c.tieWordEmbeddings then Set("lm_head.weight") else Set.empty[String]
      require((expected.keySet -- optional).subsetOf(archive.tensors.keySet) && archive.tensors.keySet.subsetOf(expected.keySet), "Llama weight names mismatch")
      archive.tensors.foreach { (name, info) =>
        require(info.shape == expected(name) && Set(SafeDType.F32, SafeDType.F16, SafeDType.BF16)(info.dtype), s"Llama weight metadata mismatch: $name")
      }
      val model = build(c, trainable, limits)((name, _) => archive.readFloat(name))
      if c.tieWordEmbeddings && archive.tensors.contains("lm_head.weight") then
        val actual = archive.readFloat("lm_head.weight").toFloatArray
        val embedding = model.embedding.weight.data.toFloatArray
        require(actual.indices.forall(i => java.lang.Float.floatToRawIntBits(actual(i)) == java.lang.Float.floatToRawIntBits(embedding(i))), "conflicting tied lm_head weight")
      model
    }
