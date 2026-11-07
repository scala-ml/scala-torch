"""Regenerate the test mapping with Diffusers' original ControlNet converter.

Usage: python scripts/reference_single_file_controlnet_names.py CHECKPOINT OUTPUT
Only the safetensors header is read; generated values are source tensor names.
This is an optional reference-fixture tool, not a runtime dependency.
"""
import json
import struct
import sys

from diffusers.loaders.single_file_utils import convert_controlnet_checkpoint

with open(sys.argv[1], "rb") as checkpoint:
    length = struct.unpack("<Q", checkpoint.read(8))[0]
    names = json.loads(checkpoint.read(length))
reference = convert_controlnet_checkpoint(
    {name: name for name in names if name != "__metadata__"}, {"layers_per_block": 2}
)
if any(value is None for value in reference.values()) or len(set(reference.values())) != len(reference):
    raise ValueError("checkpoint did not produce a one-to-one mapping")
with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as output:
    json.dump(reference, output, sort_keys=True, indent=2)
