package scalatorch.models

import scalatorch.*
import scalatorch.nn.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendTrainingSuite extends munit.FunSuite:
  override val munitTimeout=scala.concurrent.duration.Duration(5,"minutes")
  private def base(tied: Boolean = true) = LlamaModel.init(
    LlamaConfig(8,8,12,2,2,1,maxPositions=16,initializerRange=.1,tieWordEmbeddings=tied),17)
  private val input=Tensor.fromLongs(Array(1L,2L,3L,4L,4L,3L,2L,1L),Shape.from(Seq(2L,4L)))
  private val labels=Tensor.fromLongs(Array(2L,3L,4L,1L,3L,2L,1L,4L),input.shape)
  private def near(actual: Array[Float],expected: Array[Float],tolerance: Double): Unit =
    assertEquals(actual.length,expected.length)
    actual.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,tolerance*(1+math.abs(e))))
  private def gradients[T](backend: TrainingBackend[T]): Unit =
    for tied <- Seq(false,true) do
      val model=base(tied)
      val expected=model.forward(input).crossEntropy(labels)
      expected.backward()
      Using.resource(BackendLlamaTraining.prepare(model,backend)) { prepared =>
        Using.resource(prepared.newStep()) { step =>
          val logits=step.forward(input)
          val loss=logits.crossEntropy(labels)
          loss.backward()
          near(backend.download(loss.data).toFloatArray,expected.data.toFloatArray,2e-6)
          Module.namedParameters(model).foreach { (name,p) =>
            assert(step.parameter(p).requiresGrad,clues(name))
            near(backend.download(step.parameter(p).grad.get).toFloatArray,p.grad.get.toFloatArray,2e-5)
          }
          if tied then assert(step.parameter(model.embedding.weight) eq step.parameter(model.lmHead.weight))
        }
      }
  test("complete backend CPU Llama gradients match every reference parameter") { gradients(CpuFloatBackend) }
  test("complete backend CUDA Llama gradients match every reference parameter") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    CudaSession.withDevice() { session =>
      gradients(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("resident CUDA Llama SGD trains every parameter with CPU update parity") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    val model=base();val reference=base()
    val host=Module.parameters(model).map(p => p -> p.data)
    CudaSession.withDevice() { session =>
      Using.resource(BackendLlamaTraining.prepare(model,session)) { prepared =>
        val baseline=session.allocatedBytes
        var first=0f;var last=0f
        for i <- 0 until 500 do
          val rate=if i<250 then .1f else .03f
          // Compare the operation at identical weights: independent Float32 trajectories can diverge.
          Module.loadStateDict(reference,prepared.stateDict())
          val expected=reference.forward(input).crossEntropy(labels)
          expected.backward()
          Using.resource(prepared.newStep()) { step =>
            val loss=step.forward(input).crossEntropy(labels)
            loss.backward()
            val value=loss.data.toCpu.toFloatArray.head
            if i==0 then first=value
            last=value
            near(Array(value),expected.data.toFloatArray,2e-6)
            Module.parameters(model).zip(Module.parameters(reference)).foreach { (p,r) =>
              near(step.parameter(p).grad.get.toCpu.toFloatArray,r.grad.get.toFloatArray,2e-5)
            }
            step.sgd(rate)
          }
          Module.parameters(reference).foreach(p => p.load(CpuFloatBackend.add(p.data,CpuFloatBackend.scale(p.grad.get,-rate))))
          val actual=prepared.stateDict()
          Module.stateDict(reference).foreach((name,t) => near(actual(name).toFloatArray,t.toFloatArray,2e-6))
          assertEquals(session.allocatedBytes,baseline)
        assert(last<first*.25,clues(first,last))
        assert(host.forall((p,t) => p.data eq t))
        println(s"CUDA full Llama SGD loss: $first -> $last")
      }
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("CUDA SGD rejects nonfinite gradients and allocation failures without publishing weights") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    val model=base()
    CudaSession.withDevice(maxBytes=1024*1024) { session =>
      Using.resource(BackendLlamaTraining.prepare(model,session)) { prepared =>
        val original=prepared.stateDict().view.mapValues(_.toFloatArray.toVector).toMap
        Using.resource(prepared.newStep()) { step =>
          (step.forward(input).sum()*Float.NaN).backward()
          intercept[IllegalArgumentException](step.sgd(.1f))
        }
        assertEquals(prepared.stateDict().view.mapValues(_.toFloatArray.toVector).toMap,original)
        Using.resource(prepared.newStep()) { step =>
          val terms=Module.parameters(model).map(p => step.parameter(p).sum()*Float.MaxValue)
          terms.reduce(_+_).backward()
          intercept[IllegalArgumentException](step.sgd(2f))
        }
        assertEquals(prepared.stateDict().view.mapValues(_.toFloatArray.toVector).toMap,original)
        Using.resource(prepared.newStep()) { step =>
          step.forward(input).crossEntropy(labels).backward()
          val baseline=session.allocatedBytes
          val reserve=session.fill(Shape.from(Seq((session.maxBytes-baseline-600)/4)),0f)
          val reserved=session.allocatedBytes
          intercept[IllegalArgumentException](step.sgd(.1f))
          assertEquals(session.allocatedBytes,reserved)
          reserve.close()
          assertEquals(prepared.stateDict().view.mapValues(_.toFloatArray.toVector).toMap,original)
          step.sgd(.1f)
          intercept[IllegalStateException](step.sgd(.1f))
        }
      }
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("training steps preserve LoRA frozen weights and reject stale or externally changed models") {
    val adapted=LlamaLoRA.attach(base(),2,4,31,targetModules=Set("q_proj","v_proj","up_proj"))
    val model=adapted.model
    val expected=model.forward(input).crossEntropy(labels)
    expected.backward()
    Using.resource(BackendLlamaTraining.prepare(model,CpuFloatBackend)) { prepared =>
      val initial=prepared.stateDict()
      val first=prepared.newStep();val stale=prepared.newStep()
      try
        val loss=first.forward(input).crossEntropy(labels)
        loss.backward()
        Module.parameters(model).foreach { p =>
          assertEquals(first.parameter(p).requiresGrad,p.trainable)
          if p.trainable then near(first.parameter(p).grad.get.toFloatArray,p.grad.get.toFloatArray,1e-6)
          else assertEquals(first.parameter(p).grad,None)
        }
        intercept[IllegalArgumentException](first.sgd(Float.NaN))
        first.sgd(.1f)
        intercept[IllegalStateException](stale.forward(input))
        intercept[IllegalStateException](first.sgd(.1f))
        val next=prepared.stateDict()
        Module.namedParameters(model).filterNot(_._2.trainable).foreach { (name,_) =>
          assertEquals(next(name).toFloatArray.toVector,initial(name).toFloatArray.toVector)
        }
      finally first.close();stale.close()
      val live=prepared.newStep()
      intercept[IllegalArgumentException](live.sgd(.1f))
      model.norm.weight.load(model.norm.weight.data)
      intercept[IllegalArgumentException](live.forward(input))
      live.close()
    }
  }
  test("CUDA training enforces model limits and owns live steps through close") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    CudaSession.withDevice() { session =>
      val invalid=base().copy(limits=LlamaLimits(maxParameters=1))
      intercept[IllegalArgumentException](BackendLlamaTraining.prepare(invalid,session))
      assertEquals(session.allocatedBytes,0L)
      val limited=base().copy(limits=LlamaLimits(maxLogitsElements=8,maxScoreElements=1))
      val prepared=BackendLlamaTraining.prepare(limited,session)
      val step=prepared.newStep()
      intercept[IllegalArgumentException](step.forward(input))
      intercept[IllegalArgumentException](step.forward(Tensor.fromLongs(Array(1L),Shape.from(Seq(1L)))))
      intercept[IllegalArgumentException](step.forward(Tensor.fromLongs(Array(8L),Shape.from(Seq(1L,1L)))))
      intercept[IllegalArgumentException](step.forward(Tensor.fromLongs(Array(1L),Shape.from(Seq(1L,1L)))))
      val tooLong=Tensor.fromLongs(Array.fill(17)(1L),Shape.from(Seq(1L,17L)))
      intercept[IllegalArgumentException](step.forward(tooLong))
      var threadFailure: Throwable=null
      val other=new Thread(() => { try prepared.stateDict() catch case e: Throwable => threadFailure=e; () })
      other.start();other.join()
      assert(threadFailure.isInstanceOf[IllegalStateException])
      prepared.close();prepared.close();step.close()
      intercept[IllegalStateException](prepared.newStep())
      intercept[IllegalStateException](step.forward(input))
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("pretrained SmolLM2 CUDA LoRA trains resident factors with frozen base and reference gradients") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    val dir=sys.props.get("scalatorch.smollm.model.dir").getOrElse(
      assume(false,"set scalatorch.smollm.model.dir for pretrained training gate").toString)
    val original=LlamaModel.load(java.nio.file.Path.of(dir))
    val adapted=LlamaLoRA.attach(original,4,8,42,targetLayers=Some(Set(original.layers.size-1)))
    val reference=LlamaLoRA.attach(original,4,8,42,targetLayers=Some(Set(original.layers.size-1)))
    val ids=Tensor.fromLongs(Array(1L,2L,3L,4L),Shape.from(Seq(1L,4L)))
    val targets=Tensor.fromLongs(Array(2L,3L,4L,1L),ids.shape)
    val host=Module.parameters(adapted.model).map(p => p -> p.data)
    CudaSession.withDevice(maxBytes=2L*1024*1024*1024) { session =>
      Using.resource(BackendLlamaTraining.prepare(adapted.model,session)) { prepared =>
        val baseline=session.allocatedBytes
        var first=0f;var last=0f
        for i <- 0 until 12 do
          Module.parameters(reference).foreach(_.zeroGrad())
          val expected=reference.model.forward(ids).crossEntropy(targets)
          expected.backward()
          val updates=scala.collection.mutable.Map.empty[Parameter,Tensor[Float]]
          Using.resource(prepared.newStep()) { step =>
            val loss=step.forward(ids).crossEntropy(targets)
            loss.backward()
            val value=loss.data.toCpu.toFloatArray.head
            if i==0 then first=value
            last=value
            near(Array(value),expected.data.toFloatArray,2e-4)
            Module.parameters(adapted).zip(Module.parameters(reference)).foreach { (p,r) =>
              val gradient=step.parameter(p).grad.get.toCpu
              near(gradient.toFloatArray,r.grad.get.toFloatArray,2e-4)
              updates(p)=CpuFloatBackend.add(r.data,CpuFloatBackend.scale(gradient,-1f))
            }
            Module.parameters(adapted.model).filterNot(_.trainable).foreach(p => assertEquals(step.parameter(p).grad,None))
            step.sgd(1f)
          }
          // Explicit observation only: align the next CPU comparison to the resident factors.
          Using.resource(prepared.newStep()) { step =>
            Module.parameters(adapted).zip(Module.parameters(reference)).foreach { (p,r) =>
              val next=step.parameter(p).data.toCpu
              assertEquals(next.toFloatArray.toVector,updates(p).toFloatArray.toVector)
              r.load(next)
            }
            // A representative frozen table retains its original allocation and bits.
            near(step.parameter(adapted.model.norm.weight).data.toCpu.toFloatArray,adapted.model.norm.weight.data.toFloatArray,0)
          }
          assertEquals(session.allocatedBytes,baseline)
        assert(last<first*.99,clues(first,last))
        assert(host.forall((p,t) => p.data eq t))
        println(s"CUDA pretrained SmolLM2 LoRA SGD loss: $first -> $last residentBytes=$baseline")
      }
      assertEquals(session.allocatedBytes,0L)
    }
  }
