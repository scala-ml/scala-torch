# Offline trained-model text generation

The Scala CPU path loads the original pinned [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac) checkpoint and generates `The capital of France is Paris.` from the default-system chat prompt asking France's capital. Runtime requires Java25 and the existing model files, with no Python, native inference engine, conversion step or network fetch.

`TextGeneration.generate(model,tokenizer,prompt,maxNewTokens)` returns `GeneratedText(ids,text)` containing only the continuation, including EOS in IDs and skipping special tokens in text. It checks vocabulary size and requires a nonempty encoded prompt; existing generation validates IDs, requested context length and nonnegative token count. The size check cannot prove that two same-sized vocabularies match: load the model and tokenizer from the same revision. Decode the whole continuation, since an individual token can hold only part of a UTF-8 character.

`SmolChat.format(messages,addGenerationPrompt=true)` implements the inspected SmolLM2 template explicitly: default system message when the first message is not system, role/content delimiters, and optional assistant prefix. It accepts system/user/assistant roles; it does not interpret Jinja, infer model family, sanitize user content or claim support for tools. The caller chooses this formatter. The CLI accepts `DIRECTORY PROMPT [MAX_NEW_TOKENS [--smollm-chat]]`; chat mode requires an explicit count. Without the flag, input is a raw prompt.

## Verification

The pinned model has 30 layers, GQA9/3, hidden576, tied embeddings, and134515008 BF16-stored parameters. Independent Candle0.9.1 and Scala both convert weights to F32 and run on CPU. Comparison covers 49152 logits at prompt prefill and each of three following cached steps: all196608 values pass fixed `atol=rtol=2e-4`, with maximum absolute error `4.3392181396484375e-5`. All eight greedy continuation IDs match, including EOS2; both decode the same answer. Model artifacts and committed reference artifacts are SHA-256 checked by the test. See [provenance, hashes and reproduction](../../models/src/test/resources/llama/smollm2/PROVENANCE.md).

The fixture tests one prompt, not broad answer quality. Synthetic tests cover default/explicit-system formatting, history, empty history, invalid role, continuation-only output, zero new tokens, empty encoded prompt and vocabulary-size mismatch. CLI execution was also observed independently of the tests. The existing sbt client emitted a transport `InterruptedException` after the successful command/output; this is recorded separately from inference success. A following client hung; the thread dump showed the server blocked in `VirtualTerminal.cancelRequests` during channel shutdown, before tests started. After terminating those owned processes, foreground verification (`sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch ...`) exited0 with **102 passed, 0 failed, 0 skipped**, including all three public-reference gates and real CUDA.

Run the trained-model gate:

```sh
sbt -batch ';set models / Test / javaOptions ++= Seq("-Xmx3g", "-Dscalatorch.smollm.model.dir=/absolute/path/to/smollm2-135m");models / Test / testOnly scalatorch.models.SmolLlamaSuite scalatorch.models.TextGenerationSuite'
```

The model/tokenizer gate is optional in default tests and explicitly enabled during full verification. This is CPU reference inference, not a benchmark or GPU model backend. Stochastic sampling, generalized chat/tokenizer variants, model download management, sharded weights, LoRA, large-model memory planning, diffusion and the NUI migration remain open.
