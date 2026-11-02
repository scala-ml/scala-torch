"""Generate a tiny CPU DPM++ image reference; never used by Scala at runtime."""
import sys
from pathlib import Path
import torch
import transformers
import diffusers
from transformers import CLIPTokenizerFast, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor

assert diffusers.__version__ == "0.33.1"
assert transformers.__version__ == "4.49.0"
assert torch.__version__.split("+")[0] == "2.7.1"
torch.set_num_threads(1)
root, output = map(Path, sys.argv[1:])
tokenizer = CLIPTokenizerFast(vocab_file=str(root / "tokenizer/vocab.json"),
                            merges_file=str(root / "tokenizer/merges.txt"), model_max_length=77)
clip = CLIPTextModel.from_pretrained(root / "text_encoder", local_files_only=True,
                                    attn_implementation="eager").eval()
unet = UNet2DConditionModel.from_pretrained(root / "unet", local_files_only=True).eval()
vae = AutoencoderKL.from_pretrained(root / "vae", local_files_only=True).eval()
unet.set_attn_processor(AttnProcessor())
vae.set_attn_processor(AttnProcessor())
scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085,
    beta_end=0.012, beta_schedule="scaled_linear", algorithm_type="dpmsolver++",
    solver_order=2, solver_type="midpoint", use_karras_sigmas=True)
scheduler.set_timesteps(3)
with torch.no_grad():
    ids = tokenizer(["a red cat", "a blue boat"], padding="max_length", return_tensors="pt").input_ids
    positive, negative = clip(ids).last_hidden_state.split(1)
    latent = torch.tensor([(i % 7 - 3) * 0.125 for i in range(60)]).reshape(1, 4, 3, 5)
    for t in scheduler.timesteps:
        pos = unet(latent, t, encoder_hidden_states=positive).sample
        neg = unet(latent, t, encoder_hidden_states=negative).sample
        latent = scheduler.step(neg + 3 * (pos - neg), t, latent).prev_sample
    image = vae.decode(latent / vae.config.scaling_factor).sample
output.write_bytes(image.numpy().astype("<f4").tobytes())
print("shape", tuple(image.shape), "range", float(image.min()), float(image.max()))
