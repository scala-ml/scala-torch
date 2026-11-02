# Seeded token sampling

`SamplingConfig(temperature=0,topK=0,topP=1,seed=0)` is the default. Temperature0 preserves greedy decoding with the lowest token ID winning ties; filters are ignored in greedy mode. Positive temperature enables categorical sampling. `topK=0` keeps all vocabulary entries and larger-than-vocabulary values clamp to the vocabulary size. `topP` must be in `(0,1]`.

For stochastic decoding, `TokenSampler.probabilities` subtracts the largest logit in Double precision before temperature division and exponentiation. It retains the highest top-k entries first, then the smallest descending-probability prefix whose cumulative mass reaches top-p, including the crossing token. Probabilities are renormalized after filtering. Equal logits, including positive and negative zero, are ordered by ascending token ID. `topP=1` disables nucleus truncation, including for tiny nonzero tail probabilities. Empty/nonfinite logits and invalid configurations are rejected. Extreme negative exponents may underflow to zero normally.

`new TokenSampler(config).sample(logits)` owns one Java Random stream. Each `generateIds` call creates a fresh sampler, so the same seed/policy/model/input replays the same draw sequence. Use separate samplers per request. This does not promise PyTorch/Candle seed equivalence, cross-backend bitwise generation, or resumable RNG state. Numerical changes around a cumulative threshold can change selected IDs. There is no shared global sampling RNG.

```scala
import scalatorch.models.*
import java.nio.file.Path

val directory = Path.of("/absolute/path/to/smollm2-135m")
val model = LlamaModel.load(directory)
val tokenizer = ByteBpeTokenizer.read(directory.resolve("tokenizer.json"))
val prompt = SmolChat.format(Vector(ChatMessage("user", "What is the capital of France?")))
val policy = SamplingConfig(temperature=0.7,topK=20,topP=0.9,seed=42)
val answer = TextGeneration.generate(model,tokenizer,prompt,32,sampling=policy)
println(answer.text)
```

The token-ID API accepts the same optional policy: `model.generateIds(promptIds,32,useCache=true,sampling=policy)`. Existing calls retain greedy behavior; the minimal TextRun CLI remains greedy.

Tests cover analytic softmax/temperature distributions, top-k-before-top-p ordering, exact nucleus boundaries, tiny probabilities, extreme finite Float logits/tiny temperature, invalid configurations, 20000 fixed-seed categorical draws, cached/uncached model replay, and text-API policy forwarding. The trained SmolLM2 gate also repeats seeded sampling on real weights, while retaining its independent Candle greedy/logit checks. These are correctness checks, not sampling-quality or throughput benchmarks. The reference implementation sorts vocabulary indices and allocates a dense probability array each step; optimized selection, repetition penalties, batched generation and RNG-state persistence remain open.

With the documented SmolLM2 prompt and policy (temperature0.7, topK20, topP0.9, seed42), two runs produced identical IDs `504,3575,282,4649,314,7042,30,2` and text `The capital of France is Paris.` This observation demonstrates replay on the trained model; it is not an independent stochastic reference or quality benchmark. The analytical and empirical tests check the sampling distribution itself.

Full verification at this checkpoint: **108 passed, 0 failed, 0 skipped**, including all public-reference gates and real CUDA. The existing Java25 Scala-runtime Unsafe warning remains.
