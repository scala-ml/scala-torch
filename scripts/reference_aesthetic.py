"""torch / transformers oracle for the Scala aesthetic scorers (AestheticModelsSuite).

usage: python reference_aesthetic.py <weights/raw dir> <out dir> <image> [<image> ...]
<raw dir> is the download layout described in convert_aesthetic_to_safetensors.py (original checkpoints, not the converted files).
For every image writes <out>/aesthetic_ref_<stem>.safetensors with
  clip_pixels (3,224,224)       Pillow bicubic shorter-side resize + torchvision centre crop + CLIP mean/std (as clip.load's preprocess)
  clip_embed (768,)             HF CLIPModel.get_image_features of those pixels (un-normalised, ViT-L/14)
  improved (1,)                 LAION improved predictor (MLP on the L2-normalised embedding) -- the reference MLP code is re-written here
  sim_pixels_l14 / sim_embed_l14 / sim_l14 and sim_pixels_b16 / sim_embed_b16 / sim_b16   Simulacra models (Lanczos resize like rank_images.py)
  cafe_pixels (3,384,384)       transformers BeitImageProcessor output (resize 384x384 bilinear, mean/std 0.5)
  cafe_aesthetic / cafe_style / cafe_waifu   logits of BeitForImageClassification (cafeai) and <name>_probs softmax
Only needs torch, torchvision, transformers, Pillow, safetensors.
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import CLIPModel, BeitForImageClassification, BeitImageProcessor
from safetensors.torch import save_file

raw, out = sys.argv[1:3]; images = sys.argv[3:]
os.makedirs(out, exist_ok=True)
torch.set_grad_enabled(False)
norm = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

def prep(img, mode, size=224):
    return norm(TF.to_tensor(TF.center_crop(TF.resize(img, size, mode), (size, size))))

clip_l = CLIPModel.from_pretrained(os.path.join(raw, "clip")).eval()
clip_b = CLIPModel.from_pretrained(os.path.join(raw, "clipb16")).eval()

mlp = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2), nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
sd = torch.load(os.path.join(raw, "improved", "sac+logos+ava1-l14-linearMSE.pth"), map_location="cpu", weights_only=True)
mlp.load_state_dict({k.replace("layers.", ""): v for k, v in sd.items()}); mlp.eval()

def sim_head(tag, d):
    m = nn.Linear(d, 1)
    sd = torch.load(os.path.join(raw, "sim", f"sac_public_2022_06_29_{tag}_linear.pth"), map_location="cpu", weights_only=True)
    m.load_state_dict({k.replace("linear.", ""): v for k, v in sd.items()}); return m
sim = {"l14": sim_head("vit_l_14", 768), "b16": sim_head("vit_b_16", 512)}
cafe = {n: BeitForImageClassification.from_pretrained(os.path.join(raw, f"cafe_{n}")).eval() for n in ["aesthetic", "style", "waifu"]}
proc = BeitImageProcessor.from_pretrained(os.path.join(raw, "cafe_style"))

for path in images:
    img = Image.open(path).convert("RGB"); r = {}
    px = prep(img, transforms.InterpolationMode.BICUBIC)
    emb = clip_l.get_image_features(pixel_values=px[None])
    r["clip_pixels"] = px; r["clip_embed"] = emb[0]
    r["improved"] = mlp(F.normalize(emb, dim=-1))[0]
    for tag, model in [("l14", clip_l), ("b16", clip_b)]:
        p = prep(img, transforms.InterpolationMode.LANCZOS)
        e = model.get_image_features(pixel_values=p[None])
        x = F.normalize(F.normalize(e, dim=-1), dim=-1) * e.shape[-1] ** 0.5  # rank_images.py normalises the embedding, the model normalises again
        r[f"sim_pixels_{tag}"] = p; r[f"sim_embed_{tag}"] = e[0]; r[f"sim_{tag}"] = sim[tag](x)[0]
    cp = proc(img, return_tensors="pt")["pixel_values"]
    r["cafe_pixels"] = cp[0]
    for n, m in cafe.items():
        lg = m(pixel_values=cp).logits[0]; r[f"cafe_{n}"] = lg; r[f"cafe_{n}_probs"] = lg.softmax(-1)
    stem = os.path.splitext(os.path.basename(path))[0]
    save_file({k: v.float().contiguous() for k, v in r.items()}, os.path.join(out, f"aesthetic_ref_{stem}.safetensors"))
    print(stem, "improved", round(r["improved"].item(), 4), "sim_l14", round(r["sim_l14"].item(), 4), "sim_b16", round(r["sim_b16"].item(), 4),
          {n: [round(v, 3) for v in r[f"cafe_{n}_probs"].tolist()] for n in cafe})
