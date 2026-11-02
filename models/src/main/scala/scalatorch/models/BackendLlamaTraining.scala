package scalatorch.models

import scalatorch.*
import scalatorch.autograd.BackendTape
import scalatorch.nn.*
import scala.collection.mutable
import scala.util.Using

/** Resident training snapshot. Host Parameters remain unchanged; stateDict explicitly downloads.
  * A fresh scoped Step owns each graph. Plain SGD stages all replacements before publication.
  */
final class BackendLlamaTraining[T] private (model: LlamaModel,val backend: TrainingBackend[T],
    weights: mutable.Map[Parameter,T],resources: BackendScope[T]) extends AutoCloseable:
  private val owner=Thread.currentThread()
  private val named=Module.namedParameters(model)
  private val parameters=Module.parameters(model)
  private val snapshot=parameters.map(p => p -> p.value)
  private val steps=mutable.LinkedHashSet.empty[Step]
  private var generation=0L
  private var closed=false
  private val c=model.config
  private def threadCheck(): Unit =
    if Thread.currentThread() ne owner then throw IllegalStateException("training Llama used from another thread")
  private def check(): Unit =
    threadCheck()
    if closed then throw IllegalStateException("training Llama is closed")
    require(snapshot.forall((p,v) => p.value eq v),"host model weights changed; prepare again")
  def stateDict(): Map[String,Tensor[Float]] =
    check()
    val downloaded=parameters.map(p => p -> backend.download(weights(p))).toMap
    named.map((name,p) => name -> downloaded(p)).toMap
  def newStep(): Step =
    check()
    val step=new Step(generation)
    steps+=step;step

  final class Step private[BackendLlamaTraining] (version: Long) extends AutoCloseable:
    val tape=new BackendTape(backend)
    private var finished=false
    private val leaves=try parameters.map(p => p -> tape.variable(weights(p),p.trainable)).toMap
      catch case error: Throwable =>
        try tape.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
    private def valid(): Unit =
      check()
      if finished then throw IllegalStateException("training step is closed")
      if version!=generation then throw IllegalStateException("training step is stale after a weight update")
    /** Borrowed tape value; tied names resolve to the same leaf. */
    def parameter(p: Parameter): tape.Value =
      valid()
      require(leaves.contains(p),"parameter does not belong to this model")
      leaves(p)
    private def linear(input: tape.Value,layer: Linear): tape.Value =
      val base=input.matmul(leaves(layer.weight).transpose(0,1))
      layer.lora.fold(base) { a =>
        base + input.matmul(leaves(a.a).transpose(0,1)).matmul(leaves(a.b).transpose(0,1)) *
          (a.alpha/a.a.data.shape.dims(0)).toFloat
      }
    private def norm(input: tape.Value,layer: RMSNorm): tape.Value = input.rmsNorm(leaves(layer.weight),layer.epsilon)
    def forward(tokens: Tensor[Long]): tape.Value =
      valid()
      require(tokens.shape.rank==2,"Llama tokens must have shape (batch,tokens)")
      val b=tokens.shape.dims(0);val t=tokens.shape.dims(1)
      require(t<=c.maxPositions,"Llama context limit exceeded")
      val output=Shape.from(Seq(b,t,c.vocabSize.toLong))
      require(output.numel<=model.limits.maxLogitsElements && output.numel<=Int.MaxValue,"Llama logits allocation limit exceeded")
      val hiddenShape=Shape.from(Seq(b,t,c.hiddenSize.toLong))
      val matrixShape=Shape.from(Seq(b*t,c.hiddenSize.toLong))
      var hidden=leaves(model.embedding.weight).embedding(tokens)
      model.layers.foreach { layer =>
        val flat=norm(hidden,layer.inputNorm).reshape(matrixShape)
        def heads(projection: Linear,count: Int): tape.Value =
          linear(flat,projection).reshape(Shape.from(Seq(b,t,count.toLong,c.headDim.toLong))).transpose(1,2)
        val q=heads(layer.attention.qProj,c.numAttentionHeads).rope(theta=c.ropeTheta)
        val k=heads(layer.attention.kProj,c.numKeyValueHeads).rope(theta=c.ropeTheta)
        val v=heads(layer.attention.vProj,c.numKeyValueHeads)
        val attended=q.causalAttention(k,v,maxScoreElements=model.limits.maxScoreElements)
          .transpose(1,2).reshape(matrixShape)
        hidden=hidden+linear(attended,layer.attention.oProj).reshape(hiddenShape)
        val post=norm(hidden,layer.postNorm).reshape(matrixShape)
        val product=linear(post,layer.mlp.gateProj).silu()*linear(post,layer.mlp.upProj)
        hidden=hidden+linear(product,layer.mlp.downProj).reshape(hiddenShape)
      }
      linear(norm(hidden,model.norm).reshape(matrixShape),model.lmHead).reshape(output)

    /** Plain Float32 SGD. Allocation/numeric rejection leaves every resident weight unchanged.
      * Successful publication invalidates all existing steps, including this one.
      */
    def sgd(learningRate: Float): Unit =
      valid()
      require(learningRate.isFinite && learningRate>0,"invalid SGD learning rate")
      require(generation<Long.MaxValue,"training generation overflow")
      val trainable=parameters.filter(_.trainable)
      require(trainable.nonEmpty,"no trainable parameters")
      val gradients=trainable.map { p =>
        val g=leaves(p).grad.getOrElse(throw IllegalArgumentException("missing parameter gradient; run backward first"))
        require(backend.allFinite(g),"nonfinite parameter gradient")
        p -> g
      }
      Using.resource(new BackendScope(backend)) { staged =>
        val replacements=gradients.map { (p,g) =>
          val delta=staged.keep(backend.scale(g,-learningRate))
          val next=staged.keep(backend.add(weights(p),delta))
          staged.release(delta)
          require(backend.allFinite(next),"nonfinite updated parameter")
          p -> next
        }
        // All device allocation and numeric checks precede publication.
        val old=replacements.map((p,_) => weights(p))
        replacements.foreach((p,next) => weights(p)=resources.keep(staged.take(next)))
        generation+=1
        old.foreach(resources.release)
      }
    def close(): Unit =
      threadCheck()
      if !finished then
        finished=true
        try tape.close() finally steps-=this

  def close(): Unit =
    threadCheck()
    if !closed then
      closed=true
      Using.Manager { use =>
        use(resources)
        steps.toVector.foreach(step => use(step))
        ()
      }.get

object BackendLlamaTraining:
  def prepare[T](model: LlamaModel,backend: TrainingBackend[T]): BackendLlamaTraining[T] =
    require(model.layers.size==model.config.numLayers && model.layers.size<=model.limits.maxLayers,"Llama layer count/limit mismatch")
    val projections=model.layers.flatMap { l => Vector(l.attention.qProj,l.attention.kProj,l.attention.vProj,l.attention.oProj,
      l.mlp.gateProj,l.mlp.upProj,l.mlp.downProj) } :+ model.lmHead
    require(projections.forall(_.bias.isEmpty),"training Llama currently requires bias-free projections")
    val parameters=Module.parameters(model)
    require(parameters.map(p => BigInt(p.data.shape.numel)).sum<=model.limits.maxParameters,"Llama parameter limit exceeded")
    val resources=new BackendScope(backend)
    try
      val weights=mutable.Map.from(parameters.map(p => p -> resources.keep(backend.upload(p.data))))
      new BackendLlamaTraining(model,backend,weights,resources)
    catch case error: Throwable =>
      try resources.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
      throw error
