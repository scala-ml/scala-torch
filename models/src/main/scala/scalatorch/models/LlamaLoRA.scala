package scalatorch.models

import scalatorch.nn.*
import java.util.IdentityHashMap

/** Adapter-only parameter tree; model contains the frozen base plus these adapters. */
final class LlamaLoRA private (val model: LlamaModel, val adapters: Vector[(String, LowRankAdapter)])

object LlamaLoRA:
  given Module[LlamaLoRA] with
    def namedParameters(value: LlamaLoRA): Vector[(String, Parameter)] =
      value.adapters.flatMap((name, a) => Vector(s"$name.lora_a" -> a.a, s"$name.lora_b" -> a.b))

  def attach(base: LlamaModel, rank: Int, alpha: Double, seed: Long,
      targetModules: Set[String] = Set("q_proj", "v_proj"), targetLayers: Option[Set[Int]] = None,
      maxAdapterParameters: Long = 10000000L): LlamaLoRA =
    val allowed = Set("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    val selected = targetLayers.getOrElse(base.layers.indices.toSet)
    require(targetModules.nonEmpty && targetModules.subsetOf(allowed), "invalid LoRA target modules")
    require(base.layers.size == base.config.numLayers && selected.nonEmpty && selected.forall(base.layers.indices.contains),
      "invalid LoRA target layers")
    val scale = (alpha / rank).toFloat
    require(rank > 0 && alpha.isFinite && alpha > 0 && scale.isFinite && scale > 0, "invalid LoRA rank/scale")
    def projections(layer: LlamaBlock) = Vector(
      "self_attn.q_proj" -> layer.attention.qProj, "self_attn.k_proj" -> layer.attention.kProj,
      "self_attn.v_proj" -> layer.attention.vProj, "self_attn.o_proj" -> layer.attention.oProj,
      "mlp.gate_proj" -> layer.mlp.gateProj, "mlp.up_proj" -> layer.mlp.upProj, "mlp.down_proj" -> layer.mlp.downProj)
    val all = base.layers.zipWithIndex.flatMap((layer, i) => projections(layer).map((n,p) => (i,n,p)))
    require(base.lmHead.lora.isEmpty && all.forall(_._3.lora.isEmpty), "model already has adapters")
    val targets = all.filter((i,n,_) => selected(i) && targetModules(n.split('.').last))
    val count = targets.map { (_,_,p) =>
      val dims = p.weight.data.shape.dims
      require(dims.forall(d => d > 0 && BigInt(d) * rank <= Int.MaxValue), "adapter factor allocation limit exceeded")
      (BigInt(dims(0)) + dims(1)) * rank
    }.sum
    val baseCount = Module.parameters(base).map(p => BigInt(p.data.shape.numel)).sum
    require(maxAdapterParameters >= 0 && count <= maxAdapterParameters && baseCount + count <= base.limits.maxParameters,
      "LoRA parameter allocation budget exceeded")

    val frozen = new IdentityHashMap[Parameter, Parameter]()
    def freeze(p: Parameter): Parameter =
      if !frozen.containsKey(p) then frozen.put(p, Parameter(p.data, false))
      frozen.get(p)
    val adapters = Vector.newBuilder[(String, LowRankAdapter)]
    val random = new java.util.Random(seed)
    def linear(p: Linear, name: String, enabled: Boolean): Linear =
      val adapter = if enabled then
        val a = LowRankAdapter.init(p.weight.data.shape.dims(1).toInt, p.weight.data.shape.dims(0).toInt, rank, alpha, random.nextLong())
        adapters += name -> a
        Some(a)
      else None
      Linear(freeze(p.weight), p.bias.map(freeze), adapter)
    def norm(n: RMSNorm) = RMSNorm(freeze(n.weight), n.epsilon)
    val layers = base.layers.zipWithIndex.map { (layer,i) =>
      def project(p: Linear, group: String, name: String) =
        linear(p, s"model.layers.$i.$group.$name", selected(i) && targetModules(name))
      LlamaBlock(norm(layer.inputNorm), LlamaAttention(
        project(layer.attention.qProj,"self_attn","q_proj"), project(layer.attention.kProj,"self_attn","k_proj"),
        project(layer.attention.vProj,"self_attn","v_proj"), project(layer.attention.oProj,"self_attn","o_proj")),
        norm(layer.postNorm), LlamaMLP(project(layer.mlp.gateProj,"mlp","gate_proj"),
          project(layer.mlp.upProj,"mlp","up_proj"), project(layer.mlp.downProj,"mlp","down_proj")))
    }
    val model = LlamaModel(base.config, Embedding(freeze(base.embedding.weight)), layers, norm(base.norm),
      linear(base.lmHead,"lm_head",false), base.limits)
    new LlamaLoRA(model, adapters.result())
