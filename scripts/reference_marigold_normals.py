"""Reference for the Scala Marigold-normals port: runs diffusers MarigoldNormalsPipeline step by step (FP32, CUDA, TF32 off)
with an explicit initial noise latent and dumps every intermediate as raw little-endian float32.

usage: python -I reference_marigold_normals.py <marigold-normals-v1-1 dir> <image.png> <out dir> [steps=4] [res=768] [seed=0] [device=cuda]

Writes into <out dir>: noise.f32 (1x4xhxw), text.f32 (1x2x1024), image_latent.f32 (scaled), pred_latent.f32 (final target latent),
decoded.f32 (1x3xPHxPW, clipped+normalised, padding removed), normals.f32 (1x3xHxW, resized back to the input size and renormalised)
and meta.json (shapes, timesteps). Written only for verification; it is not used at runtime."""
import json, sys, os
import numpy as np
import torch
from PIL import Image
from diffusers import MarigoldNormalsPipeline

model_dir, image_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
res = int(sys.argv[5]) if len(sys.argv) > 5 else 768
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
device = sys.argv[7] if len(sys.argv) > 7 else "cuda"
os.makedirs(out_dir, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True

pipe = MarigoldNormalsPipeline.from_pretrained(model_dir, torch_dtype=torch.float32).to(device)
pipe.set_progress_bar_config(disable=True)
image = Image.open(image_path).convert("RGB")

# stage-by-stage, mirroring MarigoldNormalsPipeline.__call__
proc = pipe.image_processor
tensor, padding, original = proc.preprocess(image, res, "bilinear", torch.device(device), torch.float32)
image_latent, _ = pipe.prepare_latents(tensor, None, None, 1, 1)
g = torch.Generator("cpu").manual_seed(seed)
noise = torch.randn(image_latent.shape, generator=g, dtype=torch.float32).to(device)

ids = pipe.tokenizer("", padding="do_not_pad", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
text = pipe.text_encoder(ids)[0]

pred = noise.clone()
pipe.scheduler.set_timesteps(steps, device=device)
timesteps = [int(t) for t in pipe.scheduler.timesteps]
with torch.no_grad():
    for t in pipe.scheduler.timesteps:
        eps = pipe.unet(torch.cat([image_latent, pred], dim=1), t, encoder_hidden_states=text, return_dict=False)[0]
        pred = pipe.scheduler.step(eps, t, pred).prev_sample
    decoded = pipe.decode_prediction(pred)
decoded = proc.unpad_image(decoded, padding)
final = torch.nn.functional.interpolate(decoded, original, mode="bilinear", antialias=False)
final = pipe.normalize_normals(final)

def dump(name, t):
    t.detach().float().cpu().numpy().astype("<f4").tofile(os.path.join(out_dir, name))

dump("noise.f32", noise); dump("text.f32", text); dump("image_latent.f32", image_latent); dump("pred_latent.f32", pred)
dump("decoded.f32", decoded); dump("normals.f32", final)
json.dump({"image": image_path, "steps": steps, "res": res, "seed": seed, "timesteps": timesteps,
           "latent_shape": list(image_latent.shape), "text_shape": list(text.shape), "decoded_shape": list(decoded.shape),
           "original": list(original), "padding": list(padding),
           "scheduler": {"alphas_cumprod_head": [float(x) for x in pipe.scheduler.alphas_cumprod[:3]],
                         "alphas_cumprod_tail": [float(x) for x in pipe.scheduler.alphas_cumprod[-3:]]}},
          open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
n = ((final[0].permute(1, 2, 0) + 1) * 0.5 * 255).to(torch.uint8).cpu().numpy()
Image.fromarray(n).save(os.path.join(out_dir, "normals_ref.png"))
print("done", timesteps, tuple(image_latent.shape), tuple(decoded.shape))
