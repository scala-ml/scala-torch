# Offline text training CLI

`scalatorch.models.TrainRun` runs CPU or CUDA AdamW training from a local model directory and UTF-8 text file. It uses native tokenization, resident model steps and atomic checkpoints. It never downloads assets or invokes Python.

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch ';set models / Compile / run / javaOptions += "-Xmx3g";models / Compile / runMain scalatorch.models.TrainRun cuda /absolute/path/to/smollm2-135m /absolute/path/to/corpus.txt /absolute/path/to/run.stb 30 --sequence-length 8 --learning-rate 0.01'
```

Resume to a total of60updates with the same training settings:

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch ';set models / Compile / run / javaOptions += "-Xmx3g";models / Compile / runMain scalatorch.models.TrainRun cuda /absolute/path/to/smollm2-135m /absolute/path/to/corpus.txt /absolute/path/to/run.stb 60 --sequence-length 8 --learning-rate 0.01 --resume'
```

Use `cpu` instead of `cuda` for CPU execution. The final positional number is the desired global step, not the number of additional updates. A fresh run rejects an existing checkpoint path; `--resume` requires an existing compatible file. `--help` prints the accepted options.

## Options

| Option | Default | Meaning |
|---|---:|---|
| `--sequence-length` |64| Tokens per packed sequence; must fit model context and logits limits. |
| `--batch-size` |1| Number of sequences per update. |
| `--learning-rate` |0.001| AdamW learning rate; betas0.9/0.999, epsilon1e-8, weight decay0.01. |
| `--full` |off| Train every model parameter instead of attaching LoRA. |
| `--lora-rank` |4| Adapter rank. |
| `--lora-alpha` |8| Adapter scale numerator. |
| `--lora-layers` |`last`| `last`, `all`, or unique comma-separated zero-based layer indices. Targets q/v projections. |
| `--seed` |42| Adapter initialization seed. |
| `--save-every` |10| Checkpoint interval in global updates; completion also saves. |
| `--device-bytes` |2147483648| CUDA session allocation budget. |
| `--resume` |off| Restore weights, moments and global cursor from the checkpoint. |

`--full` rejects explicit LoRA options. Full-model AdamW requires substantially more device memory than LoRA; the budget is enforced rather than falling back to CPU. Model/tokenizer vocabulary, context, logits and adapter limits are checked before training. The supported tokenizer remains the existing byte-level BPE subset; arbitrary model families and chat templates are not inferred.

## Data and resume semantics

Each nonempty line is a document. Documents are tokenized without automatic BOS or chat formatting, then receive the configured first EOS ID unless already ending with it. They are concatenated and split into fixed-length next-token sequences. Every token transition appears once per epoch. Final input padding uses EOS; padded targets use `-100` and contribute no cross-entropy loss. EOS marks document boundaries; it does not create an attention reset inside a packed sequence.

Batch order is sequential and repeats after the corpus is exhausted. The saved global step selects the next batch, so a resumed run follows the same order. There is currently no shuffle, data augmentation, assistant-only masking or chat SFT formatting. Logs contain the loss before each update and the number of supervised target tokens. Losses from different batches are not directly comparable as a quality metric.

Input bounds are8MiB of UTF-8 text,1,048,576 resulting tokens and at most that many positions per batch. The tokenizer's existing per-input and per-piece limits also apply. Invalid UTF-8, empty/insufficient corpora, unsupported documents or exceeded limits reject rather than silently truncating training data.

The checkpoint stores a training context containing the actual token-corpus fingerprint, tokenizer-file SHA-256, batch/sequence settings, and training mode/adapter selection/initialization. CLI resume requires this context to match. Changing corpus paths is allowed when their tokenized content is identical; changing supervision, tokenizer or batching is rejected. Model architecture and optimizer configuration remain checked by the checkpoint loader. Total target steps, checkpoint interval and device budget may change.

`STBADM02` stores this bounded context explicitly and incorporates it in the architecture fingerprint. Ordinary `loadCheckpoint(path)` can validate and restore model/optimizer state without the original corpus; a training caller uses `loadCheckpoint(path, context = Some(expected))` for strict run matching. Existing `STBADM01` checkpoints remain readable. The model directory and matching AdamW configuration are still required by the current API. A convenient trained-artifact inference command remains future work.

Checkpoints include all base weights, even for LoRA. Save frequency therefore trades recovery granularity against full-model I/O. Only completed saves are recoverable after interruption; concurrent writers to one destination are unsupported. The format uses checked staging and atomic replacement as documented in [resident AdamW](backend-adamw.md).

## Evidence

Dataset tests check exact transition coverage, ignored tail padding, epoch rollover and maximum Long step selection, defensive input copying, stable fingerprints, malformed UTF-8 and size/token/dimension bounds. CPU CLI tests train both full models and LoRA; the CUDA CLI test trains LoRA. Four uninterrupted updates and2+2resumed updates produce byte-identical checkpoints. Changed data and accidental checkpoint overwrite reject without altering the saved file. Context corruption, portable loading and legacy format compatibility are covered separately.

Separate forked JVM commands also ran pinned SmolLM2-135M-Instruct against a four-line text corpus. CPU and CUDA first-step loss were both **5.6415625**. A fresh CUDA process resumed from step1 and saved step2, whose different batch had loss **6.411142**. This verifies the command workflow; it is not a convergence or language-quality claim. Local evidence is `.superpowers/training-cli-native.log`; corpus/checkpoint artifacts remain ignored under `.superpowers/training-cli/`.

Next work includes trained-artifact inference ergonomics, richer data formats and sampling policies, and a Transformers-style library above the current model layer. Optimized kernels/memory handling, diffusion and complete Python elimination from NUI remain part of the overall objective.

## Inference from a training checkpoint

Run generation without the original training corpus or optimizer arguments:

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch 'models / Compile / runMain scalatorch.models.BackendTextRun cuda MODEL_DIRECTORY "What is the capital of France?" 16 --smollm-chat --checkpoint saved.stb'
```

Use `cpu` for CPU execution. The model directory still supplies the original config, safetensors and tokenizer. `TrainedLlama.load(directory, checkpoint)` reconstructs full-model or LoRA settings from TrainRun metadata, checks the exact tokenizer-file SHA-256 and restores saved weights. It validates optimizer settings, moment values/counters and the complete checksum while streaming past moments without retaining their arrays. Inference does not allocate a resident optimizer.

For metadata-free API checkpoints and legacy STBADM01 files, use `TrainedLlama.restore(matchingModel, checkpoint)`; callers supply the matching full/LoRA architecture. Automatic reconstruction requires TrainRun metadata because adapter scaling cannot be inferred from weight shapes.

Tests delete the corpus before inference, compare restored weights/logits/greedy tokens against the training restore path on CPU/CUDA, and reject changed tokenizer bytes. A separate forked CUDA CLI loaded the actual two-step SmolLM2 LoRA checkpoint and emitted `The capital of France is Paris.` This proves saved-artifact generation, not improved model quality. Existing calls without `--checkpoint` continue loading the published model.
