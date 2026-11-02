package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

/** Trainable NCHW grouped convolution with OIHW weights and optional output-channel bias. */
final case class Conv2d(weight: Parameter,bias: Option[Parameter] = None,
    config: Conv2dConfig = Conv2dConfig()) derives Module:
  require(weight.data.shape.rank==4 && weight.data.shape.dims.forall(n => n>0 && n<=Int.MaxValue),
    "Conv2d weight must have positive OIHW dimensions")
  require(weight.data.shape.dims.head%config.groups==0,"Conv2d output channels must divide into groups")
  bias.foreach(p => require(p.data.shape==Shape.from(Seq(weight.data.shape.dims.head)),"Conv2d bias shape mismatch"))
  def apply(input: Variable): Variable =
    val output=input.conv2d(weight.value,config)
    bias.fold(output)(p => output+p.value.reshape(Shape.from(Seq(1L,weight.data.shape.dims.head,1L,1L))))
