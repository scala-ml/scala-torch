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

CPU Float32 only. No dropout, merged export, multiple adapters per projection or PEFT format compatibility. Use the bound checkpoint API below for adapter training persistence. Generic `Checkpoint` alone does not bind alpha or base identity.

## Bound adapter checkpoints

```scala
import scalatorch.models.LoRACheckpoint
val path = java.nio.file.Path.of("adapter.stl")
LoRACheckpoint.save(adapted, optimizer, path, step=30)
// Recreate the same base, targets, rank, alpha and optimizer settings first.
val savedStep = LoRACheckpoint.load(adapted, optimizer, path)
```

The loader restores A/B weights and optimizer state, clears gradients, and returns the caller-supplied step. The checkpoint contains no base tensor payload. Recreate the base and adapter configuration yourself; the initialization seed of the fresh adapter need not match. Saving/loading requires an optimizer boundary and exclusive access to model/optimizer state. RNG, data-loader state and mid-accumulation gradients are not serialized.

Format `STLORA01` is a deliberately small envelope over the existing [training checkpoint](checkpoint-format.md):

| Field | Encoding |
| --- | --- |
| Magic | 8 ASCII bytes `STLORA01` |
| Binding | 32-byte SHA-256 of the canonical base/configuration encoding |
| Payload length | nonnegative big-endian signed 64-bit integer |
| Payload | exactly that many bytes, one `STCKPT01` adapter-only checkpoint |

No trailing bytes are accepted. `maxBytes` bounds the complete envelope (default 1GiB). Loading copies the bounded payload to a private temporary file before the existing checkpoint loader validates its checksum, shapes, optimizer configuration, membership and moments. All checks precede any parameter/optimizer publication. Saving stages in the destination directory and requires atomic replacement; failure leaves the existing destination intact. Temporary files are cleaned on success and failure.

The binding covers all Llama config fields, actual RMSNorm epsilons, structural names and aliases of every frozen base parameter (including biases), shapes, raw Float32 bits, adapter target names, factor dimensions and exact Double alpha. Canonical encoding is versioned by `scalatorch-llama-lora-base-v1`: length-prefixed UTF-8 text and big-endian primitives, sorted base alias groups, then deterministic attachment-order adapters. Model allocation limits are execution policy and excluded. The hash is recomputed on every save/load to detect changed frozen parameter snapshots; hashing a large base is consequently not free. Checksums detect corruption and mismatches, not maliciously rewritten files or source authenticity.

Tests cover exact tiny-Llama AdamW continuation, wrong base/alpha/config rejection, corrupt/truncated/trailing payloads and excessive length, preservation of existing gradients and optimizer continuation after rejection, and failed-save preservation/cleanup. The public SmolLM2 gate saves the trained 7,680-parameter adapter in less than 200KB, reloads into different adapter initialization, compares logits exactly, and verifies identical parameters after the next AdamW step. This is the library's training format, not PEFT or safetensors adapter interchange.
