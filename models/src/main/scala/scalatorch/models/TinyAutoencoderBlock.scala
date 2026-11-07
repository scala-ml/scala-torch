package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.{Conv2d,Module}

/** Shape-preserving residual block used by the TAESD encoder and decoder. */
final case class TinyAutoencoderBlock(conv1: Conv2d,conv2: Conv2d,conv3: Conv2d) derives Module:
  private val channels=conv1.weight.data.shape.dims.head
  private val expected=Shape.from(Seq(channels,channels,3L,3L))
  require(Vector(conv1,conv2,conv3).forall(c => c.weight.data.shape==expected &&
    c.config==Conv2dConfig(padding=(1,1))),"tiny autoencoder block requires shape-preserving 3x3 convolutions")

  def apply(input: Variable): Variable =
    require(input.data.shape.rank==4 && input.data.shape.dims(1)==channels,
      "tiny autoencoder block input channels mismatch")
    (conv3(conv2(conv1(input).relu()).relu())+input).relu()
