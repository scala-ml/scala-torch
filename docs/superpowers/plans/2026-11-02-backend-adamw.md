# Resident AdamW and continuation

Execute inline under the approved GPU training plan.

1. Define immutable AdamW configuration and independently owned backend moment state, using Double moments to match the existing CPU optimizer. Exports/restores are explicit host transfers.
2. Establish CPU/CUDA update and moment reference tests, small/extreme gradients, numeric/shape/counter validation, ownership, exact budgets and exact continuation after state restore.
3. Implement Double CPU arithmetic and native CUDA update/status kernels, stage owned weight/state outputs, and clean all rejected allocations or nonfinite results without mutating input state.
4. Integrate persistent moments with resident Llama steps; publish weights, moments and counters together. Verify full model and pretrained LoRA training against CPU AdamW.
5. Add validated atomic checkpoint save/load with model/configuration/alias binding and uninterrupted-versus-restored training gates. Review and run all public/CUDA checks. Training CLI, performance/memory work and diffusion/NUI remain open.
