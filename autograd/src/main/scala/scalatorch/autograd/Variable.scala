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

  /** Class-index cross entropy over the last axis; targets have the remaining shape. */
  def crossEntropy(targets: Tensor[Long], reduction: LossReduction = LossReduction.Mean, ignoreIndex: Long = -100): Variable =
    LossOps.crossEntropy(this,targets,reduction,ignoreIndex)

  def resizeNearest2d(height: Int,width: Int,mode: NearestMode = NearestMode.Floor): Variable =
    Variable.result(Resize2d.nearest(data,height,width,mode),Vector(
      this -> ((g: Tensor[Float]) => Resize2d.backward(data,g,height,width,mode))))

  /** Normalize each logical final-axis row independently using biased variance. */
  def layerNorm(epsilon: Double = 1e-5): Variable =
    require(data.shape.rank>=1 && data.shape.dims.last>0,"LayerNorm requires a positive final dimension")
    val width=data.shape.dims.last
    reshape(Shape.from(Seq(data.shape.numel/width,width))).groupNorm(1,epsilon).reshape(data.shape)

  def groupNorm(groups: Int,epsilon: Double = 1e-5): Variable =
    Variable.result(GroupNormOps.forward(data,groups,epsilon),Vector(
      this -> ((g: Tensor[Float]) => GroupNormOps.backward(data,g,groups,epsilon))))

  def pad2d(padding: Padding2d): Variable =
    Variable.result(SpatialPadding.forward(data,padding),Vector(
      this -> ((g: Tensor[Float]) => SpatialPadding.backward(data,g,padding))))

  def conv2d(weight: Variable, config: Conv2dConfig = Conv2dConfig()): Variable =
    val inputSnapshot=data
    val weightSnapshot=weight.data
    Variable.result(Conv2dOps.forward(inputSnapshot,weightSnapshot,config),Vector(
      this -> ((g: Tensor[Float]) => Conv2dOps.backward(inputSnapshot,weightSnapshot,g,config,true)),
      weight -> ((g: Tensor[Float]) => Conv2dOps.backward(inputSnapshot,weightSnapshot,g,config,false))))

  def embedding(indices: Tensor[Long]): Variable = TransformerOps.embedding(this, indices)
  def rmsNorm(weight: Variable, epsilon: Double = 1e-6): Variable = TransformerOps.rmsNorm(this, weight, epsilon)
  def gelu(approximation: GeluApproximation = GeluApproximation.Erf): Variable = GeluOps(this,approximation)
  def silu(): Variable = TransformerOps.silu(this)
  /** Noncausal scaled dot-product attention over all keys, with optional grouped query heads. */
  def scaledDotProductAttention(key: Variable,value: Variable,maxScoreElements: Long = 16777216L): Variable =
    AttentionOps.dense(this,key,value,maxScoreElements)
  def causalAttention(key: Variable, value: Variable, queryStart: Int = 0,
      maxScoreElements: Long = 16777216L): Variable = AttentionOps.causal(this, key, value, queryStart, maxScoreElements)
  def rope(positionStart: Int = 0, theta: Double = 10000.0): Variable = AttentionOps.rope(this, positionStart, theta)

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

  /** Contiguous range on one logical axis; backward scatters into the original shape. */
  def narrow(dim: Int,start: Long,length: Long): Variable =
    require(dim>=0 && dim<data.shape.rank,"narrow axis out of range")
    val size=data.shape.dims(dim)
    require(start>=0 && start<=size && length>=0 && length<=size-start,"invalid narrow bounds")
    require(data.shape.numel<=Int.MaxValue,"narrow input exceeds Int indexing")
    val selected=data.slice(dim,start,start+length)
    Variable.result(selected,Vector(this -> ((g: Tensor[Float]) =>
      val out=new Array[Float](data.shape.numel.toInt)
      if selected.shape.numel>0 then
        val suffix=Shape.from(data.shape.dims.drop(dim+1)).numel
        val selectedRow=length*suffix
        val inputRow=size*suffix
        val offset=start*suffix
        val seed=g.toFloatArray
        var i=0
        while i<seed.length do
          val target=(i/selectedRow)*inputRow+offset+i%selectedRow
          out(target.toInt)=seed(i)
          i+=1
      Tensor.fromFloats(out,data.shape))))

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

  private[autograd] def result(data: Tensor[Float], edges: Vector[(Variable, Tensor[Float] => Tensor[Float])]): Variable =
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
