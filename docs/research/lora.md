# CPU LoRA adapters

`LowRankAdapter` computes `x Aᵀ Bᵀ * alpha/rank`, with A shaped `(rank,in)` and B shaped `(out,rank)`. Initialization samples A uniformly within ±1/√in and zeros B. This implements the frozen-base low-rank update described in [Hu et al., LoRA](https://arxiv.org/abs/2106.09685).

```scala
import scalatorch.models.LlamaLoRA
import scalatorch.nn.{AdamW, Module}

val adapted = LlamaLoRA.attach(model, rank=4, alpha=8.0, seed=42,
  targetModules=Set("q_proj", "v_proj"),
  targetLayers=Some(Set(model.layers.size-1)))
val optimizer = new AdamW(Module.parameters(adapted), learningRate=.01)
optimizer.zeroGrad()
adapted.model.forward(input).crossEntropy(targets).backward()
optimizer.step()
```

`Module.parameters(adapted)` exposes only A/B parameters. `adapted.model` includes frozen base parameters and trainable adapters. Attachment creates independent parameter handles over immutable base snapshots and preserves aliases, including tied embeddings/head. Subsequent updates to the source model do not replace the adapted model's snapshots.

Targets support q/k/v/o attention projections and gate/up/down MLP projections. Layers default to all. Unknown or empty selections, existing adapters, invalid ranks/scales, oversized factors and parameter budgets reject before adapter allocation. Default adapter budget is 10 million parameters; combined base plus adapter parameters must also fit the model's limit. Updating adapters invalidates existing KV caches until reset.

Verified execution evidence:

- Analytical forward, input, A and B gradients for a nonzero scaled adapter.
- Zero-B initialization preserves outputs; AdamW leaves frozen weights/bias unchanged.
- Tiny Llama: selected q/v/down projections, rank 2, alpha 4, 50 AdamW steps at .03 reduce loss from 2.0808644 to 1.5040104. Base weights remain bitwise equal and tied aliases survive.
- The [pinned SmolLM2 model](trained-text-generation.md): final-layer q/v, rank 4, alpha 8, 7,680 trainable parameters. Input `[1,2,3,4]`, targets `[2,3,4,1]`, 30 AdamW steps at .01 reduce loss from 20.862272 to 4.6144342. All adapter gradients are finite and frozen base Tensor snapshots retain identity. This synthetic-token test validates pretrained adapter training, not useful fine-tuning quality. Six steps only reached 19.631618 and did not satisfy the chosen 10% reduction gate.

CPU Float32 only. No dropout, merged export, multiple adapters per projection or PEFT format compatibility. Adapter bundle persistence and exact optimizer resume remain pending. Generic `Checkpoint` does not bind alpha or base identity, so it is not yet a complete adapter interchange format.
