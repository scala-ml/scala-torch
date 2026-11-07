# Native standalone upscale

Implement the existing NUI `upscale` worker request using scala-torch's native
pixel upscalers. Accept the existing request fields, preserve Python's rounded
target dimensions and supported built-in/model choices, and report unsupported
model types explicitly. Keep the operation independent of
checkpoint loading. Stage the PNG before the publication barrier and remove it
on cancellation or failure. Exercise real protocol fields, fractional sizes,
invalid inputs, cancellation, and NUI's Generator path where practical.

For `R-ESRGAN 4x+`, repeat the native 4x model until the requested size is
covered (at most three passes), then resample to the final size as in NUI.
Keep bounded image allocations and use the existing CUDA execution setting.

Hardware-info requests also return best-effort CPU/GPU/RAM provenance, timing
and the optional remark; use `unknown` if the system facts cannot be collected.
