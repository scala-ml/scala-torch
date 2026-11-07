# Diffusion LoRA inference design

## Intent

Let the NUI scala-torch worker apply standard safetensors LoRA adapters during
SD1.x and SDXL generation. NUI already resolves adapter paths and sends a list
of names and multipliers; the native worker currently rejects nonempty lists.

## Goals

- Apply rank-decomposed linear deltas to CLIP and U-Net projections for SD1.5 and
  both SDXL text encoders.
- Accept standard Kohya keys (`lora_down/up.weight`) and Diffusers/PEFT keys
  (`lora_A/B.weight`) when their base layer resolves to a loaded linear weight.
- Support multiple adapters per request, independent text-encoder/U-Net scales,
  negative scales, and `alpha/rank` normalization.
- Keep base checkpoint weights immutable and make adapter state request-scoped,
  exception-safe, and safe under concurrent worker requests.
- Reject incomplete pairs, unsupported layer types/keys, nonfinite scales and
  unsafe allocation sizes before denoising.
- Keep adapter names, paths and model-specific aliases in the scala-torch
  `models` module/worker boundary. NUI UI and shared protocol types stay backend
  agnostic.

## Non-goals

- LyCORIS LoHa/LoKr, LoCon convolution factors, DoRA, IA3, dynamic rank,
  per-block weights, step schedules, hires-specific adapters, or composable
  adapter routing.
- Python checkpoint/pickle loading; this path accepts safetensors only.
- Changing the existing default Python backend or claiming full Python removal.

## Approaches

1. Merge deltas into the loaded model. This mutates or duplicates the base and
   complicates request-to-request switching, so it is not selected.
2. Clone the full model graph with adapters attached for each request. This
   preserves weights but duplicates a large object graph and complicates CUDA
   resource lifetime.
3. Use a scoped per-request adapter table keyed by base `Parameter` identity.
   Existing `Linear` and CUDA paths already implement low-rank math; an exception-
   safe scope can add adapter factors without mutating or cloning the model.
   **Chosen** for immutable base weights and per-request multiplier control.

## Constraints and acceptance

- scala-torch runtime remains Python-free and only consumes safetensors.
- A request with no adapters preserves current output exactly.
- CPU and CUDA `Linear` execution both include all scoped adapters and release
  scope state after success, failure and cancellation.
- SD1.5/SDXL safetensors fixtures cover aliases, shapes, alpha, multiple adapters,
  CLIP/U-Net scale separation, invalid pairs and deterministic replay.
- A tiny-model native worker regression proves nonzero adapters change output,
  repeated seeded output is identical, zero scale is a no-op, and base output is
  restored after the request.
- Unsupported LoRA families fail explicitly; no feature is silently ignored.
- This is a supported standard linear-LoRA slice; normal-checkpoint image
  quality, memory/performance and nonstandard adapter families remain open.
