package scalatorch.models

import scalatorch.nn.Parameter

/** A loaded model plus stable diffusion checkpoint names for its Linear weights. */
final case class LoadedLinearModel[A](model: A, linearTargets: Map[String, Parameter])
