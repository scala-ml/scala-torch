"""Export NUI's 1-based LoRA window policy for boundary-focused tests."""
import json
import sys
from pathlib import Path

nui, output = map(Path, sys.argv[1:3])
sys.path.insert(0, str(nui / "python"))
from sdfx.lora_blocks import active

cases = []
for total in (1, 2, 3, 4, 5, 8):
    for start in (None, -1.5, 0, .1, .25, .5, .75, 1, 2, 8):
        for stop in (None, -1.5, 0, .1, .25, .5, .75, 1, 2, 8):
            cases.append({"total": total, "start": start, "stop": stop,
                          "active": [active(step, total, start, stop) for step in range(1, total + 1)]})
output.write_text(json.dumps(cases), encoding="utf-8")
