# Cross-entropy and causal language-model training

`Variable.crossEntropy(targets,reduction=LossReduction.Mean,ignoreIndex = -100)` computes class-index loss from unnormalized logits. Classes occupy the **last axis**: `(batch,time,vocab)` logits take `(batch,time)` Long targets; a single `(classes)` vector takes a scalar target. This layout is chosen for the Scala Llama API rather than mirroring every PyTorch input layout.

The unweighted class-index objective is negative log-softmax at each target, consistent with the [CrossEntropyLoss definition](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html). Ignored targets contribute neither loss nor gradient. `None` returns the target shape, `Sum` totals valid losses, and `Mean` divides by the number of nonignored targets. This increment does not implement class weights, soft targets or label smoothing.

Input logits must be finite, including ignored rows. Zero classes, wrong target shapes and out-of-range nonignored target IDs reject. Empty/all-ignored Mean explicitly returns NaN with a zero gradient; Sum returns zero and None returns an empty/zero tensor as appropriate. Training callers should skip batches with no valid targets rather than interpreting their NaN mean as a usable metric. Input class count and materialized element counts must fit Int indexing.

Forward computes shifted log-sum-exp in Double. Separate per-row maximum and logarithmic-sum statistics prevent huge equal logits from losing the normalization term. Backward uses softmax minus one-hot, scaled by the reduction and upstream gradient; it recomputes probabilities from saved Float logits and those two Double row statistics, without retaining a full Double probability tensor. The Float32 loss can still overflow if the mathematical value exceeds Float32 range. The operation uses existing first-order graph lifetime/version/noGrad rules.

## Language-model training

Shift the token sequence explicitly; the loss itself does not shift or choose masks:

```scala
import scalatorch.*
import scalatorch.nn.*
import scalatorch.models.*

val model = LlamaModel.init(
  LlamaConfig(8,8,16,1,2,1,maxPositions=16,initializerRange=0.1), seed=7)
val tokens = Tensor.fromLongs(
  Array(1L,2L,3L,4L,1L, 4L,1L,2L,3L,4L), Shape.from(Seq(2L,5L)))
val input = tokens.slice(1,0,4)
val targets = tokens.slice(1,1,5)
val optimizer = new AdamW(Module.parameters(model),learningRate=0.03)
for _ <- 0 until 60 do
  optimizer.zeroGrad()
  val loss = model.forward(input).crossEntropy(targets)
  loss.backward()
  optimizer.step()
```

The test observes mean loss `2.1745813 → 0.00066011405` and all eight next-token predictions correct after60 steps. Every parameter receives finite gradients on each step. This proves a differentiable Llama/optimizer/loss path on a small synthetic transition task, not successful pretrained fine-tuning, generalization or LoRA completion. Training uses full differentiable forward; the inference-only KV cache is not used.

## Verification

Six loss tests cover analytic losses, nonuniform unreduced upstream gradients, ignore/reduction normalization, every-coordinate central differences through transposed logits and targets, huge equal logits, a singleton class, empty/all-ignored reductions, validation, noGrad and stale-graph rejection. The separate end-to-end Llama test checks optimization and predicted tokens. Default tests require no Python or model downloads. Higher derivatives, GPU loss kernels, label smoothing, weighted/probabilistic targets and distributed reduction remain open.

Full checkpoint verification: **123 passed, 0 failed, 0 skipped**, with all four public-reference gates and actual CUDA enabled. Review found no blocking defects.
