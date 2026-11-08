"""One-off offline converter for the aesthetic scorers' weights -> plain float32 safetensors (no pickle at runtime).

usage: python convert_aesthetic_to_safetensors.py <raw dir> <out dir>
<raw dir> layout (see the downloads in the notes): clip/ (openai/clip-vit-large-patch14 model.safetensors), clipb16/ (openai/clip-vit-base-patch16
pytorch_model.bin), improved/sac+logos+ava1-l14-linearMSE.pth (camenduru/improved-aesthetic-predictor mirror of christophschuhmann's file),
sim/sac_public_2022_06_29_vit_{l_14,b_16}_linear.pth (crowsonkb/simulacra-aesthetic-models), cafe_{aesthetic,style,waifu}/pytorch_model.bin
(cafeai, BEiT-base-patch16-384 classifiers; the int64 relative_position_index buffers are dropped and recomputed at load time).
Output (names the Scala loader expects): clip-vit-large-patch14-vision.safetensors, clip-vit-base-patch16-vision.safetensors (only vision_model.* and
visual_projection.weight), improved-aesthetic-sac-logos-ava1-l14-linearMSE.safetensors, simulacra_l14.safetensors, simulacra_b16.safetensors,
cafe_aesthetic.safetensors, cafe_style.safetensors, cafe_waifu.safetensors. Needs torch + safetensors.
"""
import sys, os, json, torch
from safetensors.torch import save_file, load_file

raw, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)

def save(name, sd):
    sd = {k: v.detach().float().contiguous() for k, v in sd.items()}
    save_file(sd, os.path.join(out, name))
    print(name, len(sd), "tensors", sum(v.numel() for v in sd.values()), "params")

for src, dst in [("clip", "clip-vit-large-patch14-vision.safetensors"), ("clipb16", "clip-vit-base-patch16-vision.safetensors")]:
    st, pt = os.path.join(raw, src, "model.safetensors"), os.path.join(raw, src, "pytorch_model.bin")
    sd = load_file(st) if os.path.exists(st) else torch.load(pt, map_location="cpu", weights_only=True)  # B/16 only ships pytorch_model.bin
    save(dst, {k: v for k, v in sd.items() if (k.startswith("vision_model.") or k == "visual_projection.weight") and "position_ids" not in k})

save("improved-aesthetic-sac-logos-ava1-l14-linearMSE.safetensors",
     torch.load(os.path.join(raw, "improved", "sac+logos+ava1-l14-linearMSE.pth"), map_location="cpu", weights_only=True))
for tag, f in [("l14", "vit_l_14"), ("b16", "vit_b_16")]:
    save(f"simulacra_{tag}.safetensors", torch.load(os.path.join(raw, "sim", f"sac_public_2022_06_29_{f}_linear.pth"), map_location="cpu", weights_only=True))
for c, labels in [("aesthetic", ["not_aesthetic", "aesthetic"]), ("style", ["anime", "real_life", "3d", "manga_like", "other"]), ("waifu", ["not_waifu", "waifu"])]:
    cfg = json.load(open(os.path.join(raw, f"cafe_{c}", "config.json")))
    assert [cfg["id2label"][str(i)] for i in range(len(labels))] == labels, f"cafe_{c} labels {cfg['id2label']}"
    sd = torch.load(os.path.join(raw, f"cafe_{c}", "pytorch_model.bin"), map_location="cpu", weights_only=True)
    save(f"cafe_{c}.safetensors", {k: v for k, v in sd.items() if "relative_position_index" not in k})
