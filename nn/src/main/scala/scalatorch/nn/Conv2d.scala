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
    val (effInput,effConfig)=CircularConvPadding.current match
      case Some(tiling) if !input.requiresGrad && (config.padding._1>0 || config.padding._2>0) =>
        (Variable(CircularConvPadding.pad(input.data,config.padding._1,config.padding._2,tiling.tileX,tiling.tileY),requiresGrad=false),
          config.copy(padding=(0,0)))
      case _ => (input,config)
    val base=effInput.conv2d(weight.value,effConfig)
    val output=LinearAdapterScope.forWeight(weight).foldLeft(base) { case (sum,(adapter,multiplier)) =>
      sum+adapter.convolution(effInput,effConfig)*multiplier
    }
    bias.fold(output)(p => output+p.value.reshape(Shape.from(Seq(1L,weight.data.shape.dims.head,1L,1L))))
