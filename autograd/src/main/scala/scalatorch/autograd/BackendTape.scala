package scalatorch.autograd

import scalatorch.*
import scala.collection.mutable
import scala.util.Using

/** Scoped first-order backend graph. Inputs are borrowed through independent handles.
  * Graph values and leaf gradients live until close; use a fresh tape per training step.
  * No implicit transfers, broadcasting, mutation, or higher-order gradients.
  */
final class BackendTape[T](val backend: TrainingBackend[T]) extends AutoCloseable:
  private val thread=Thread.currentThread()
  private var closed=false
  private val nodes=mutable.ArrayBuffer.empty[Value]
  private class Handles extends AutoCloseable:
    private val values=mutable.ArrayBuffer.empty[T]
    private val attention=mutable.Map.empty[AnyRef,(T,T,T)]
    def attentionGradients(key: AnyRef)(compute: => (T,T,T)): (T,T,T) =
      attention.getOrElseUpdate(key,{
        val result=compute
        keep(result._1);keep(result._2);keep(result._3)
        result
      })
    def keep(value: T): T = { values+=value;value }
    def take(value: T): T =
      val index=values.lastIndexOf(value)
      require(index>=0,"handle is not owned")
      values.remove(index);value
    def release(value: T): Unit = backend.release(take(value))
    def close(): Unit =
      var failure: Throwable=null
      values.reverseIterator.foreach { value =>
        try backend.release(value) catch case e: Throwable =>
          if failure==null then failure=e else failure.addSuppressed(e)
      }
      values.clear()
      attention.clear()
      if failure!=null then throw failure
  private val owned=new Handles
  private def check(): Unit =
    if Thread.currentThread() ne thread then throw IllegalStateException("backend tape used from another thread")
    if closed then throw IllegalStateException("backend tape is closed")
  private type Edge=(Value,(T,Handles) => T)
  private def result(data: => T,edges: Vector[Edge]): Value =
    check()
    val tracked=edges.filter(_._1.requiresGrad)
    val node=new Value(owned.keep(data),tracked.nonEmpty,tracked)
    nodes+=node;node

  /** The caller retains ownership of source. The tape keeps its own reshape handle. */
  def variable(source: T,requiresGrad: Boolean = true): Value =
    check()
    val value=new Value(owned.keep(backend.reshape(source,backend.shape(source))),requiresGrad,Vector.empty)
    nodes+=value;value

  final class Value private[BackendTape] (val data: T,val requiresGrad: Boolean,private val edges: Vector[Edge]):
    private var accumulated: Option[T]=None
    def grad: Option[T] = { check();accumulated }
    def zeroGrad(): Unit =
      check()
      accumulated.foreach(owned.release)
      accumulated=None
    def +(other: Value): Value = result(backend.add(data,other.data),Vector(
      this -> ((g,_) => backend.reshape(g,backend.shape(data))),
      other -> ((g,_) => backend.reshape(g,backend.shape(other.data)))))
    def *(other: Value): Value = result(backend.multiply(data,other.data),Vector(
      this -> ((g,_) => backend.multiply(g,other.data)),other -> ((g,_) => backend.multiply(g,data))))
    def *(scalar: Float): Value = result(backend.scale(data,scalar),Vector(this -> ((g,_) => backend.scale(g,scalar))))
    def -(other: Value): Value = this + other * -1f
    def causalAttention(key: Value,value: Value,queryStart: Int = 0,maxScoreElements: Long = 16777216L): Value =
      val identity=new Object
      def pullback(g: T,s: Handles,index: Int): T =
        val gradients=s.attentionGradients(identity)(backend.causalAttentionBackward(data,key.data,value.data,g,queryStart,maxScoreElements))
        val selected=if index==0 then gradients._1 else if index==1 then gradients._2 else gradients._3
        backend.reshape(selected,backend.shape(selected))
      result(backend.causalAttention(data,key.data,value.data,queryStart,maxScoreElements),Vector(
        this -> ((g,s) => pullback(g,s,0)),key -> ((g,s) => pullback(g,s,1)),value -> ((g,s) => pullback(g,s,2))))
    def embedding(indices: Tensor[Long]): Value = result(backend.embedding(data,indices),
      Vector(this -> ((g,_) => backend.embeddingBackward(backend.shape(data),indices,g))))
    def rope(positionStart: Int = 0,theta: Double = 10000.0): Value =
      result(backend.rope(data,positionStart,theta),Vector(this -> ((g,_) => backend.ropeBackward(g,positionStart,theta))))
    def silu(): Value = result(backend.silu(data),Vector(this -> ((g,_) => backend.siluBackward(data,g))))
    def rmsNorm(weight: Value,epsilon: Double = 1e-6): Value = result(backend.rmsNorm(data,weight.data,epsilon),Vector(
      this -> ((g,_) => backend.rmsNormInputBackward(data,weight.data,g,epsilon)),
      weight -> ((g,_) => backend.rmsNormWeightBackward(data,g,epsilon))))
    def matmul(other: Value): Value = result(backend.matmul(data,other.data),Vector(
      this -> ((g,s) => backend.matmul(g,s.keep(backend.transpose(other.data,0,1)))),
      other -> ((g,s) => backend.matmul(s.keep(backend.transpose(data,0,1)),g))))
    def transpose(dim0: Int,dim1: Int): Value = result(backend.transpose(data,dim0,dim1),
      Vector(this -> ((g,_) => backend.transpose(g,dim0,dim1))))
    def reshape(shape: Shape): Value = result(backend.reshape(data,shape),
      Vector(this -> ((g,_) => backend.reshape(g,backend.shape(data)))))
    def sum(): Value = result(backend.sum(data),Vector(this -> ((g,_) => backend.expandScalar(g,backend.shape(data)))))
    def mean(): Value =
      check()
      require(backend.shape(data).numel>0,"mean of an empty backend value")
      sum() * (1f/backend.shape(data).numel.toFloat)
    def backward(): Unit =
      check()
      require(backend.shape(data).rank==0,"implicit backward seed requires a scalar")
      val seed=backend.fill(Shape.from(Seq.empty),1f)
      Using.resource(new AutoCloseable { def close(): Unit = backend.release(seed) })(_ => backward(seed))
    def backward(seed: T): Unit =
      check()
      require(requiresGrad,"cannot differentiate an untracked output")
      require(backend.shape(seed)==backend.shape(data),"backward seed shape mismatch")
      Using.resource(new Handles) { scratch =>
        val pending=mutable.Map.empty[Value,T]
        pending(this)=scratch.keep(backend.reshape(seed,backend.shape(seed)))
        // Nodes are appended after their parents; only reachable nodes receive gradients.
        nodes.reverseIterator.foreach { node =>
          pending.get(node).foreach { g =>
            node.edges.foreach { (parent,pullback) =>
              val contribution=scratch.keep(pullback(g,scratch))
              pending(parent)=pending.get(parent).fold(contribution)(old => scratch.keep(backend.add(old,contribution)))
            }
          }
        }
        val leaves=pending.iterator.filter((node,_) => node.edges.isEmpty).map { (node,g) =>
          node -> node.accumulated.fold(g)(old => scratch.keep(backend.add(old,g)))
        }.toVector
        // No existing leaf gradients change if any backward allocation/computation fails.
        leaves.foreach { (node,g) =>
          val next=owned.keep(scratch.take(g))
          node.accumulated.foreach(owned.release)
          node.accumulated=Some(next)
        }
      }

  def close(): Unit =
    if Thread.currentThread() ne thread then throw IllegalStateException("backend tape closed from another thread")
    if !closed then
      closed=true
      try owned.close() finally nodes.clear()
