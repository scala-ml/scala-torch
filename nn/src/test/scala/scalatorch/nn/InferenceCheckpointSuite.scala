package scalatorch.nn

import scalatorch.*
import java.nio.file.Files

class InferenceCheckpointSuite extends munit.FunSuite:
  case class Model(weight: Parameter) derives Module
  test("inference weight load discovers AdamW settings and leaves the template unchanged") {
    val path=Files.createTempFile("checkpoint-inference-",".stc")
    try
      val source=Model(Parameter(Tensor.fromFloats(Array(1f,-2f),Shape.from(Seq(2L)))))
      val optimizer=new AdamW(Module.parameters(source),.037,beta1=.8,beta2=.95,epsilon=1e-6,weightDecay=.12)
      source.weight.value.backward(Tensor.fromFloats(Array(.3f,-.7f),source.weight.data.shape));optimizer.step()
      Checkpoint.save(source,optimizer,path,7)
      val template=Model(Parameter(Tensor.fromFloats(Array(9f,9f),Shape.from(Seq(2L)))))
      val loaded=BackendAdamCheckpoint.loadWeights(template,path)
      assertEquals(loaded.step,7L)
      assertEquals(loaded.weights("weight").toFloatArray.toVector,source.weight.data.toFloatArray.toVector)
      assertEquals(template.weight.data.toFloatArray.toVector,Vector(9f,9f))
      assert(loaded.moments.isEmpty)
    finally Files.deleteIfExists(path)
  }
  test("skipped moments remain numerically checked even with a valid recomputed checksum") {
    val path=Files.createTempFile("checkpoint-invalid-moment-",".stc")
    try
      val model=Model(Parameter(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L)))))
      val optimizer=new AdamW(Module.parameters(model),.01)
      model.weight.value.backward(Tensor.fromFloats(Array(.3f),model.weight.data.shape));optimizer.step()
      Checkpoint.save(model,optimizer,path,1)
      val original=Files.readAllBytes(path)
      val b=java.nio.ByteBuffer.wrap(original).order(java.nio.ByteOrder.BIG_ENDIAN)
      def text(): Unit = { val n=b.getInt();b.position(b.position()+n);() }
      b.position(16);text()
      val settings=b.getInt()
      for _ <- 0 until settings do { text();b.getDouble() }
      assertEquals(b.getInt(),1);assertEquals(b.getInt(),1);text()
      b.get();assertEquals(b.getInt(),1);assertEquals(b.getLong(),1L)
      b.getFloat();assertEquals(b.get(),1.toByte);assertEquals(b.get(),1.toByte)
      b.getLong()
      val first=b.position()
      for (offset,value) <- Seq(first -> Double.NaN,(first+8) -> -1.0) do
        val body=original.dropRight(32)
        java.nio.ByteBuffer.wrap(body).putDouble(offset,value)
        Files.write(path,body++java.security.MessageDigest.getInstance("SHA-256").digest(body))
        val before=model.weight.data
        intercept[IllegalArgumentException](BackendAdamCheckpoint.loadWeights(model,path))
        assert(model.weight.data eq before)
    finally Files.deleteIfExists(path)
  }
