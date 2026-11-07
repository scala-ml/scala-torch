"""Verify the archive produced by SafeTensorsWriterSuite with upstream safetensors."""
from safetensors import safe_open
import sys

with safe_open(sys.argv[1],framework="np") as f:
    assert f.metadata()=={"format":"pt","日本語":"猫"}
    assert f.get_tensor("float").tolist()==[[1.,4.],[2.,5.],[3.,6.]]
    assert f.get_tensor("half").tolist()==[1.,-2.,2**-24,65504.]
    assert f.get_tensor("double").tolist()==[1.25,-2.5]
    assert f.get_tensor("long").tolist()==[9007199254740993,-7]
    assert f.get_tensor("bool").tolist()==[False,True]
    assert f.get_tensor("empty").shape==(0,3)
    assert f.get_slice("bfloat").get_shape()==[3]
with safe_open(sys.argv[1],framework="pt") as f:
    assert f.get_tensor("bfloat").float().tolist()==[1.,-2.,.5]
print("upstream safetensors verified")
