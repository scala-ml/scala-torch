package scalatorch.autograd

import scalatorch.*
import scalatorch.cpu.NumscalaBackend
import scala.collection.mutable
import java.util.IdentityHashMap

/** First-order CPU Float32 differentiable value. Leaf gradients accumulate until zeroGrad. */
final class Variable private (
    val data: Tensor[Float], val requiresGrad: Boolean,
    private[autograd] val edges: Vector[(Variable, Tensor[Float] => Tensor[Float])],
    private val dependencies: Vector[Variable]
):
  private var accumulated: Option[Tensor[Float]] = None
  private var current = true
  def grad: Option[Tensor[Float]] = accumulated
  def zeroGrad(): Unit = accumulated = None
  def detach: Variable = Variable(data, requiresGrad = false)

  /** Parameter updates replace immutable leaves and invalidate graphs saving the old value. */
  private[scalatorch] def replaced(next: Tensor[Float]): Variable =
    require(edges.isEmpty && dependencies.isEmpty, "only leaves can be replaced")
    require(next.shape == data.shape, "parameter replacement shape mismatch")
    val replacement = Variable(next, requiresGrad)
    replacement.accumulated = accumulated
    current = false
    replacement

  def +(other: Variable): Variable =
    Variable.result(Ops.add(data, other.data), Vector(
      this -> ((g: Tensor[Float]) => Variable.unbroadcast(g, data.shape)),
      other -> ((g: Tensor[Float]) => Variable.unbroadcast(g, other.data.shape))))

  def *(other: Variable): Variable =
    Variable.result(Variable.multiply(data, other.data), Vector(
      this -> ((g: Tensor[Float]) => Variable.unbroadcast(Variable.multiply(g, other.data), data.shape)),
      other -> ((g: Tensor[Float]) => Variable.unbroadcast(Variable.multiply(g, data), other.data.shape))))

  def *(scalar: Float): Variable = this * Variable.scalar(scalar)
  def -(other: Variable): Variable = this + other * -1f

  def matmul(other: Variable): Variable =
    Variable.result(NumscalaBackend.matmul(data, other.data), Vector(
      this -> ((g: Tensor[Float]) => NumscalaBackend.matmul(g, other.data.transpose(0, 1))),
      other -> ((g: Tensor[Float]) => NumscalaBackend.matmul(data.transpose(0, 1), g))))

  def sum(): Variable = sum(data.shape.dims.indices, keepDim = false)
  def sum(dims: Seq[Int], keepDim: Boolean = false): Variable =
    val output = Ops.sum(data, dims, keepDim)
    val axes = dims.toSet
    val expanded = Shape.from(data.shape.dims.zipWithIndex.map((n, i) => if axes(i) then 1L else n))
    Variable.result(output, Vector(this -> ((g: Tensor[Float]) =>
      g.contiguous.reshape(expanded).broadcastTo(data.shape))))

  def mean(): Variable =
    require(data.shape.numel > 0, "mean of empty Variable is unsupported")
    sum() * (1f / data.shape.numel.toFloat)

  def relu(): Variable =
    val source = data.toFloatArray
    val value = Tensor.fromFloats(source.map(x => math.max(0f, x)), data.shape)
    Variable.result(value, Vector(this -> ((g: Tensor[Float]) =>
      val upstream = g.toFloatArray
      var i = 0
      while i < upstream.length do
        if !(source(i) > 0) then upstream(i) = 0f
        i += 1
      Tensor.fromFloats(upstream, data.shape))))

  def transpose(dim0: Int, dim1: Int): Variable =
    Variable.result(data.transpose(dim0, dim1), Vector(this -> ((g: Tensor[Float]) => g.transpose(dim0, dim1))))
  def reshape(shape: Shape): Variable =
    // Differentiable reshape may copy a noncontiguous input; raw Tensor.reshape stays strict.
    Variable.result(data.contiguous.reshape(shape), Vector(this -> ((g: Tensor[Float]) => g.contiguous.reshape(data.shape))))
  def broadcastTo(shape: Shape): Variable =
    Variable.result(data.broadcastTo(shape), Vector(this -> ((g: Tensor[Float]) => Variable.unbroadcast(g, data.shape))))

  def backward(): Unit =
    require(data.shape.rank == 0, "implicit backward seed requires a scalar output")
    backward(Tensor.fromFloats(Array(1f), data.shape))

  /** Graph retained for reuse; gradients are detached and only leaves retain .grad. Not concurrent on shared leaves. */
  def backward(seed: Tensor[Float]): Unit =
    require(requiresGrad, "cannot differentiate an untracked output")
    require(seed.shape == data.shape, "backward seed shape differs from output")
    val seen = new IdentityHashMap[Variable, java.lang.Boolean]()
    val order = mutable.ArrayBuffer.empty[Variable]
    val stack = mutable.ArrayDeque((this, false))
    while stack.nonEmpty do
      val (node, expanded) = stack.removeLast()
      if expanded then order += node
      else if !seen.containsKey(node) then
        require(node.current, "a parameter saved by this graph has changed; run forward again")
        seen.put(node, true)
        stack.append((node, true))
        node.dependencies.foreach(parent => stack.append((parent, false)))
    val adjoints = new IdentityHashMap[Variable, Tensor[Float]]()
    adjoints.put(this, seed)
    order.reverseIterator.foreach { node =>
      val gradient = adjoints.get(node)
      if gradient != null then node.edges.foreach { (parent, pullback) =>
        val contribution = pullback(gradient)
        val previous = adjoints.get(parent)
        adjoints.put(parent, if previous == null then contribution else Ops.add(previous, contribution))
      }
    }
    // Publish after all pullbacks succeed so a rejected backward does not partially update leaves.
    order.iterator.filter(leaf => leaf.requiresGrad && leaf.edges.isEmpty).foreach { leaf =>
      val gradient = adjoints.get(leaf)
      if gradient != null then
        leaf.accumulated = Some(leaf.accumulated.fold(gradient)(Ops.add(_, gradient)))
    }

object Variable:
  def apply(data: Tensor[Float], requiresGrad: Boolean = true): Variable =
    require(data.device == Device.Cpu && data.dtype == DType.Float32, "autograd requires CPU Float32")
    new Variable(data, requiresGrad, Vector.empty, Vector.empty)

  private def scalar(value: Float): Variable =
    Variable(Tensor.fromFloats(Array(value), Shape.from(Seq.empty)), requiresGrad = false)

  private def result(data: Tensor[Float], edges: Vector[(Variable, Tensor[Float] => Tensor[Float])]): Variable =
    val tracked = if Autograd.recording then edges.filter(_._1.requiresGrad) else Vector.empty
    // Frozen transforms still save parameter lineage for version checks. Explicit noGrad/detach are snapshots.
    new Variable(data, tracked.nonEmpty, tracked, if Autograd.recording then edges.map(_._1) else Vector.empty)

  private def unbroadcast(gradient: Tensor[Float], target: Shape): Tensor[Float] =
    val padding = gradient.shape.rank - target.rank
    require(padding >= 0, "invalid broadcast pullback rank")
    val padded = Vector.fill(padding)(1L) ++ target.dims
    val axes = padded.indices.filter(i => i < padding || (padded(i) == 1 && gradient.shape.dims(i) != 1))
    Ops.sum(gradient, axes, keepDim = true).contiguous.reshape(target)

  private def multiply(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] =
    // Use the same broadcast shape rules as addition, without constructing an extra output Tensor.
    val rank = math.max(a.shape.rank, b.shape.rank)
    val ad = Vector.fill(rank - a.shape.rank)(1L) ++ a.shape.dims
    val bd = Vector.fill(rank - b.shape.rank)(1L) ++ b.shape.dims
    val shape = Shape.from(ad.zip(bd).map { (x, y) =>
      require(x == y || x == 1 || y == 1, "incompatible broadcast dimensions")
      if x == 1 then y else x
    })
    val av = a.broadcastTo(shape).toFloatArray
    val bv = b.broadcastTo(shape).toFloatArray
    var i = 0
    while i < av.length do
      av(i) *= bv(i)
      i += 1
    Tensor.fromFloats(av, shape)
