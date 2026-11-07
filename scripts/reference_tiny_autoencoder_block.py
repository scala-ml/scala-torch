"""Generate a deterministic Diffusers AutoencoderTinyBlock reference vector."""
import torch
from diffusers.models.unets.unet_2d_blocks import AutoencoderTinyBlock

block = AutoencoderTinyBlock(2, 2, 'relu').eval()
with torch.no_grad():
    for index, layer in enumerate((block.conv[0], block.conv[2], block.conv[4])):
        layer.weight.copy_((torch.arange(layer.weight.numel()).reshape_as(layer.weight) % 11 - 5) * (0.005 * (index + 1)))
        layer.bias.copy_(torch.tensor([0.02, -0.03]) * (index + 1))
    x = (torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3) - 5) / 4
    print(block(x).flatten().tolist())
