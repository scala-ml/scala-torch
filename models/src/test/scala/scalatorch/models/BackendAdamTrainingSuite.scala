package scalatorch.models

import scalatorch.*
import scalatorch.nn.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendAdamTrainingSuite extends munit.FunSuite:
  override val munitTimeout=scala.concurrent.duration.Duration(5,"minutes")
  private def base=LlamaModel.init(LlamaConfig(8,8,12,2,2,1,maxPositions=8,tieWordEmbeddings=true,initializerRange=.1),17)
  private val ids=Tensor.fromLongs(Array(1L,2L,3L,4L),Shape.from(Seq(1L,4L)))
  private val labels=Tensor.fromLongs(Array(2L,3L,4L,1L),ids.shape)
  private val config=AdamWConfig(.03,weightDecay=.01)
  private def train[T](backend: AdamWBackend[T]): Unit =
    val model=base;val reference=base
    val optimizer=new AdamW(Module.parameters(reference),config.learningRate,config.beta1,config.beta2,config.epsilon,config.weightDecay)
    Using.resource(BackendLlamaTraining.prepareAdamW(model,backend,config)) { prepared =>
      var first=0f;var last=0f
      for i <- 0 until 30 do
        optimizer.zeroGrad()
        val expected=reference.forward(ids).crossEntropy(labels)
        expected.backward()
        Using.resource(prepared.newStep()) { step =>
          val loss=step.forward(ids).crossEntropy(labels)
          loss.backward()
          val value=backend.download(loss.data).toFloatArray.head
          if i==0 then first=value
          last=value
          assertEqualsDouble(value,expected.data.toFloatArray.head,2e-5)
          Module.parameters(model).zip(Module.parameters(reference)).foreach { (p,r) =>
            val g=backend.download(step.parameter(p).grad.get)
            g.toFloatArray.zip(r.grad.get.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,2e-5*(1+math.abs(e))))
            // Separate update arithmetic from allowed CPU/GPU gradient rounding.
            r.zeroGrad();r.value.backward(g)
          }
          intercept[IllegalArgumentException](step.sgd(.1f))
          step.adamW()
          intercept[IllegalStateException](step.adamW())
        }
        optimizer.step()
        val actual=prepared.stateDict()
        Module.stateDict(reference).foreach((name,t) => assertEquals(actual(name).toFloatArray.toVector,t.toFloatArray.toVector))
        assert(prepared.optimizerState().values.forall(_.step==i+1))
      assert(last<first*.1,clues(first,last))
      println(s"Backend AdamW Llama loss: $first -> $last")
    }
  test("resident CPU Llama AdamW matches existing optimizer and reduces loss") { train(CpuFloatBackend) }
  test("resident CUDA Llama AdamW matches existing optimizer and reduces loss") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    CudaSession.withDevice() { session => train(session);assertEquals(session.allocatedBytes,0L) }
  }
  test("pretrained CUDA SmolLM2 LoRA AdamW matches reference and resumes its checkpoint exactly") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    val directory=sys.props.get("scalatorch.smollm.model.dir").getOrElse(
      assume(false,"set scalatorch.smollm.model.dir for pretrained AdamW gate").toString)
    val original=LlamaModel.load(java.nio.file.Path.of(directory))
    def attach(seed: Long) = LlamaLoRA.attach(original,4,8,seed,targetLayers=Some(Set(original.layers.size-1)))
    val adapted=attach(42);val reference=attach(42)
    val settings=AdamWConfig(.01)
    val optimizer=new AdamW(Module.parameters(reference),settings.learningRate,settings.beta1,settings.beta2,settings.epsilon,settings.weightDecay)
    val path=java.nio.file.Files.createTempFile("smollm-backend-adam-",".stb")
    def slotValues[T](p: BackendLlamaTraining[T]) = p.optimizerState().view.mapValues(s =>
      (s.step,s.first.toDoubleArray.toVector,s.second.toDoubleArray.toVector)).toMap
    try CudaSession.withDevice(maxBytes=2L*1024*1024*1024) { session =>
      var first=0f;var last=0f
      val expected=Using.resource(BackendLlamaTraining.prepareAdamW(adapted.model,session,settings)) { prepared =>
        val baseline=session.allocatedBytes
        val stateBytes=Module.parameters(adapted).map(_.data.shape.numel).sum*16
        for i <- 0 until 30 do
          optimizer.zeroGrad()
          val cpuLoss=reference.model.forward(ids).crossEntropy(labels)
          cpuLoss.backward()
          Using.resource(prepared.newStep()) { step =>
            val loss=step.forward(ids).crossEntropy(labels)
            loss.backward()
            val value=loss.data.toCpu.toFloatArray.head
            if i==0 then first=value
            last=value
            assertEqualsDouble(value,cpuLoss.data.toFloatArray.head,2e-4*(1+math.abs(value)))
            Module.parameters(adapted).zip(Module.parameters(reference)).foreach { (p,r) =>
              val gradient=step.parameter(p).grad.get.toCpu
              gradient.toFloatArray.zip(r.grad.get.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,2e-4*(1+math.abs(e))))
              r.zeroGrad();r.value.backward(gradient)
            }
            Module.parameters(adapted.model).filterNot(_.trainable).foreach(p => assertEquals(step.parameter(p).grad,None))
            step.adamW()
          }
          optimizer.step()
          Using.resource(prepared.newStep()) { step =>
            Module.parameters(adapted).zip(Module.parameters(reference)).foreach { (p,r) =>
              assertEquals(step.parameter(p).data.toCpu.toFloatArray.toVector,r.data.toFloatArray.toVector)
            }
          }
          assertEquals(session.allocatedBytes,baseline+stateBytes)
          if i==14 then prepared.saveCheckpoint(path,15)
        assert(last<first*.5,clues(first,last))
        val trained=Using.resource(prepared.newStep()) { step =>
          Module.namedParameters(adapted.model).filter(_._2.trainable).map { (name,p) =>
            name -> step.parameter(p).data.toCpu.toFloatArray.toVector
          }.toMap
        }
        (trained,slotValues(prepared))
      }
      assertEquals(session.allocatedBytes,0L)
      val fresh=attach(99)
      Using.resource(BackendLlamaTraining.prepareAdamW(fresh.model,session,settings)) { prepared =>
        assertEquals(prepared.loadCheckpoint(path),15L)
        val baseline=session.allocatedBytes
        for _ <- 0 until 15 do
          Using.resource(prepared.newStep()) { step =>
            step.forward(ids).crossEntropy(labels).backward();step.adamW()
          }
          assertEquals(session.allocatedBytes,baseline)
        Using.resource(prepared.newStep()) { step =>
          Module.namedParameters(fresh.model).filter(_._2.trainable).foreach { (name,p) =>
            assertEquals(step.parameter(p).data.toCpu.toFloatArray.toVector,expected._1(name))
          }
        }
        assertEquals(slotValues(prepared),expected._2)
        println(s"CUDA SmolLM2 LoRA AdamW loss: $first -> $last; exact checkpoint continuation; residentBytes=$baseline")
      }
      assertEquals(session.allocatedBytes,0L)
    } finally java.nio.file.Files.deleteIfExists(path)
  }
