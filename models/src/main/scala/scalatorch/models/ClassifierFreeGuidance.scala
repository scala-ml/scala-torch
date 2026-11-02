package scalatorch.models

import scalatorch.autograd.Variable

object ClassifierFreeGuidance:
  def combine(unconditional: Variable,conditional: Variable,scale: Float): Variable =
    require(scale.isFinite && scale>=0,"invalid guidance scale")
    require(unconditional.data.shape==conditional.data.shape,"guidance branch shape mismatch")
    unconditional+(conditional-unconditional)*scale
