package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

/** Stride-two image convolution, with symmetric or explicit bottom/right padding. */
final case class DiffusionDownsample2d(conv: Conv2d) derives Module:
  require(conv.weight.data.shape.dims.drop(2)==Vector(3L,3L) &&
    (conv.config==Conv2dConfig(stride=(2,2),padding=(1,1)) || conv.config==Conv2dConfig(stride=(2,2))),
    "downsampling requires a 3x3 stride-two convolution with padding zero or one")
  def apply(input: Variable): Variable =
    val padded=if conv.config.padding==(0,0) then input.pad2d(Padding2d(0,1,0,1)) else input
    conv(padded)

/** Explicit output size supports odd-size U-Net skip connections. */
final case class DiffusionUpsample2d(conv: Conv2d) derives Module:
  require(conv.weight.data.shape.dims.drop(2)==Vector(3L,3L) && conv.config==Conv2dConfig(padding=(1,1)),
    "upsampling requires a 3x3 stride-one convolution with padding one")
  def apply(input: Variable,height: Int,width: Int): Variable =
    conv(input.resizeNearest2d(height,width))
