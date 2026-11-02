package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.Variable

/** Stable parameter identity; Tensor snapshots remain immutable across updates. */
final class Parameter(initial: Tensor[Float], val trainable: Boolean = true):
  private var leaf = Variable(initial, requiresGrad = trainable)
  def value: Variable = leaf
  def data: Tensor[Float] = leaf.data
  def grad: Option[Tensor[Float]] = leaf.grad
  def zeroGrad(): Unit = leaf.zeroGrad()
  private[nn] def update(next: Tensor[Float]): Unit = leaf = leaf.replaced(next)
  def load(next: Tensor[Float]): Unit =
    update(next)
    zeroGrad()
