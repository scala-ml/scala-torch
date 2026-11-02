package scalatorch.models

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import java.nio.file.Files
import scala.util.Using

class BackendAdamCheckpointSuite extends munit.FunSuite:
  private def model(seed: Long) = LlamaModel.init(LlamaConfig(8,8,12,1,2,1,maxPositions=8,tieWordEmbeddings=true,initializerRange=.1),seed)
  private val ids=Tensor.fromLongs(Array(1L,2L,3L),Shape.from(Seq(1L,3L)))
  private val labels=Tensor.fromLongs(Array(2L,3L,1L),ids.shape)
  private val config=AdamWConfig(.02)
  private def run[T](prepared: BackendLlamaTraining[T],steps: Int): Unit =
    for _ <- 0 until steps do Using.resource(prepared.newStep()) { step =>
      step.forward(ids).crossEntropy(labels).backward();step.adamW()
    }
  private def weights[T](p: BackendLlamaTraining[T]) = p.stateDict().view.mapValues(_.toFloatArray.toVector).toMap
  private def moments[T](p: BackendLlamaTraining[T]) = p.optimizerState().view.mapValues(s =>
    (s.step,s.first.toDoubleArray.toVector,s.second.toDoubleArray.toVector)).toMap
  private def continuation[T](backend: AdamWBackend[T]): Unit =
    val path=Files.createTempFile("backend-adam-",".stb")
    try
      val expected=Using.resource(BackendLlamaTraining.prepareAdamW(model(17),backend,config)) { original =>
        run(original,7);original.saveCheckpoint(path,7)
        run(original,13)
        (weights(original),moments(original))
      }
      Using.resource(BackendLlamaTraining.prepareAdamW(model(99),backend,config)) { resumed =>
        val stale=resumed.newStep()
        try
          assertEquals(resumed.loadCheckpoint(path),7L)
          intercept[IllegalStateException](stale.forward(ids))
        finally stale.close()
        run(resumed,13)
        assertEquals(weights(resumed),expected._1)
        assertEquals(moments(resumed),expected._2)
      }
    finally Files.deleteIfExists(path)
  test("CPU AdamW checkpoint resumes full Llama weights and moments exactly") { continuation(CpuFloatBackend) }
  test("CUDA AdamW checkpoint resumes full Llama weights and moments exactly") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    CudaSession.withDevice() { session => continuation(session);assertEquals(session.allocatedBytes,0L) }
  }
  test("CUDA AdamW update and checkpoint restore roll back partial staging and can retry") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    val path=Files.createTempFile("backend-adam-budget-",".stb")
    try CudaSession.withDevice(maxBytes=256*1024) { session =>
      Using.resource(BackendLlamaTraining.prepareAdamW(model(17),session,config)) { prepared =>
        run(prepared,1);prepared.saveCheckpoint(path,1)
        val oldWeights=weights(prepared);val oldMoments=moments(prepared)
        Using.resource(prepared.newStep()) { step =>
          step.forward(ids).crossEntropy(labels).backward()
          val baseline=session.allocatedBytes
          val reserve=session.fill(Shape.from(Seq((session.maxBytes-baseline-3000)/4)),0f)
          val reserved=session.allocatedBytes
          intercept[IllegalArgumentException](step.adamW())
          assertEquals(session.allocatedBytes,reserved)
          assertEquals(weights(prepared),oldWeights);assertEquals(moments(prepared),oldMoments)
          reserve.close();step.adamW()
        }
        assert(prepared.optimizerState().values.forall(_.step==2))
        val beforeWeights=weights(prepared);val beforeMoments=moments(prepared)
        val baseline=session.allocatedBytes
        val reserve=session.fill(Shape.from(Seq((session.maxBytes-baseline-3000)/4)),0f)
        val reserved=session.allocatedBytes
        intercept[IllegalArgumentException](prepared.loadCheckpoint(path))
        assertEquals(session.allocatedBytes,reserved)
        assertEquals(weights(prepared),beforeWeights);assertEquals(moments(prepared),beforeMoments)
        reserve.close()
        assertEquals(prepared.loadCheckpoint(path),1L)
        assertEquals(weights(prepared),oldWeights);assertEquals(moments(prepared),oldMoments)
        assertEquals(session.allocatedBytes,baseline)
      }
      assertEquals(session.allocatedBytes,0L)
    } finally Files.deleteIfExists(path)
  }
  test("checkpoint rejects corruption, configuration and byte-budget mismatches without publication") {
    val path=Files.createTempFile("backend-adam-invalid-",".stb")
    try Using.resource(BackendLlamaTraining.prepareAdamW(model(17),CpuFloatBackend,config)) { prepared =>
      run(prepared,1);prepared.saveCheckpoint(path,1)
      val original=Files.readAllBytes(path)
      val originalWeights=weights(prepared);val originalMoments=moments(prepared)
      intercept[IllegalArgumentException](prepared.saveCheckpoint(path,2,maxBytes=48))
      assertEquals(Files.readAllBytes(path).toVector,original.toVector)
      intercept[IllegalArgumentException](prepared.loadCheckpoint(path,maxBytes=original.length-1))
      val corrupt=original.clone();corrupt(corrupt.length-1)=(corrupt.last^1).toByte
      for bytes <- Seq(corrupt,original.take(40),original.dropRight(10),original++Array(0.toByte)) do
        Files.write(path,bytes)
        intercept[IllegalArgumentException](prepared.loadCheckpoint(path))
        assertEquals(weights(prepared),originalWeights);assertEquals(moments(prepared),originalMoments)
      Files.write(path,original)
      val source=model(99)
      val wrong=source.copy(config=source.config.copy(ropeTheta=20000))
      Using.resource(BackendLlamaTraining.prepareAdamW(wrong,CpuFloatBackend,config)) { other =>
        intercept[IllegalArgumentException](other.loadCheckpoint(path))
      }
      Using.resource(BackendLlamaTraining.prepareAdamW(source,CpuFloatBackend,config.copy(learningRate=.1))) { other =>
        intercept[IllegalArgumentException](other.loadCheckpoint(path))
      }
    } finally Files.deleteIfExists(path)
  }
  test("saving nonfinite initial weights cannot replace an existing checkpoint") {
    val path=Files.createTempFile("backend-adam-nonfinite-",".stb")
    try
      Using.resource(BackendLlamaTraining.prepareAdamW(model(17),CpuFloatBackend,config))(_.saveCheckpoint(path,0))
      val saved=Files.readAllBytes(path).toVector
      val bad=model(17)
      bad.norm.weight.load(Tensor.fromFloats(Array.fill(8)(Float.NaN),bad.norm.weight.data.shape))
      Using.resource(BackendLlamaTraining.prepareAdamW(bad,CpuFloatBackend,config)) { prepared =>
        intercept[IllegalArgumentException](prepared.saveCheckpoint(path,0))
      }
      assertEquals(Files.readAllBytes(path).toVector,saved)
    finally Files.deleteIfExists(path)
  }
