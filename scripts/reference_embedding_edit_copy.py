"""Generate a NUI-produced byte-preserving editor fixture, not used at runtime."""
from pathlib import Path
import hashlib
import sys
import torch
from safetensors.torch import save_file

nui=Path(sys.argv[1]);sys.path.insert(0,str(nui/"python"))
from sdfx.embedding_editor_safetensors import build_safetensors_copy
out=Path("io/src/test/resources/safetensors")
src=out/"edit-source.safetensors"
dst=out/"edit-expected.safetensors"
values={"half":torch.tensor([[1.,-2.],[.25,10.]],dtype=torch.float16),
        "bfloat":torch.tensor([[.5,-4.],[1.25,2.]],dtype=torch.bfloat16),
        "single":torch.tensor([3.,-5.],dtype=torch.float32),
        "double":torch.tensor([[1.25,-2.5]],dtype=torch.float64),
        "aux":torch.tensor([4,-7,19],dtype=torch.int64)}
save_file(values,src,metadata={"format":"pt","note":"unchanged metadata"})
baseline={name:t.float().reshape(-1,t.shape[-1]).tolist() for name,t in values.items() if name!="aux"}
edited={name:[row.copy() for row in rows] for name,rows in baseline.items()}
edited["half"][1][1]=9.5;edited["bfloat"][0][1]=-3.5
edited["single"][0][0]=3.25;edited["double"][0][1]=-2.25
dst.unlink(missing_ok=True)
build_safetensors_copy(src,dst,edited,hashlib.sha256(src.read_bytes()).hexdigest(),baseline=baseline)
for path in (src,dst): print(path,hashlib.sha256(path.read_bytes()).hexdigest())
