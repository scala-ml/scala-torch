"""Check Scala-produced latent PNGs against NUI's Python unpacker and fixtures."""
import pathlib
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, 'C:/Users/Mizushima/dev/stablediffusion-nui/python')
from sdfx.vae_tools import unpack

folder, fixtures = map(pathlib.Path, sys.argv[1:])
for channels in (4, 16):
    with Image.open(folder / f'native-{channels}.png') as image:
        assert image.mode == 'RGBA'
        decoded = unpack(image, 'RGBA', channels).reshape(-1)
    expected = np.fromfile(fixtures / f'rgba-{channels}.f32', dtype='<f4')
    assert np.array_equal(decoded, expected)
    print(f'channels={channels} exact={np.array_equal(decoded, expected)}')
with Image.open(folder / 'transparent.png') as image:
    assert image.mode == 'RGBA' and image.tobytes() == bytes((255, 0, 166, 0))
    decoded = unpack(image, 'RGBA', 4).reshape(-1)
    assert decoded[0] == 10 and decoded[1] == -10 and decoded[3] == -10
    print('transparent RGB preserved')
