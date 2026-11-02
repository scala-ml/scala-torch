package scalatorch.models

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.concurrent.duration.*
import scala.util.Using

class BackendLlamaSuite extends munit.FunSuite:
  override val munitTimeout=5.minutes
  private val config=LlamaConfig(8,8,16,2,2,1,maxPositions=16,initializerRange=.1)
  private val input=Tensor.fromLongs(Array(1L,2L,3L,4L,3L,2L),Shape.from(Seq(2L,3L)))
  private def compare(actual: Array[Float],expected: Array[Float],tolerance: Double): Double =
    assertEquals(actual.length,expected.length)
    var maximum=0.0
    actual.zip(expected).foreach { (a,e) =>
      val error=math.abs(a.toDouble-e)
      assert(error<=tolerance*(1+math.abs(e.toDouble)),clues(a,e,error))
      maximum=math.max(maximum,error)
    }
    maximum
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
  private def resource(path: String): Array[Byte] =
    val stream=getClass.getResourceAsStream(path)
    assert(stream!=null)
    val bytes=try stream.readAllBytes() finally stream.close()
    val hashes=Map(
      "/llama/tiny-candle-f32.bin" -> "9c94f848e6203198bd0d469cb529d4130aca4f8c7d53a44d8ceca917212cf2c4",
      "/llama/smollm2/ids.txt" -> "456ccb39f252eedc58f2286798457bba2c1bcd43fb736fbda554000c35c0f626",
      "/llama/smollm2/logits-f32.bin" -> "3e72506c0e7d8bb67a2601a18931566a17c20ee1fa14979ccd16b9a447f4d5f4",
      "/llama/smollm2/text.txt" -> "a1b7eb2ee7a6aded8dda4e6cf30826f5afffb28a5597ee9389e91eb326d4e319")
    assertEquals(java.security.MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString,hashes(path))
    bytes
  private def sha(path: java.nio.file.Path): String =
    val digest=java.security.MessageDigest.getInstance("SHA-256")
    val stream=new java.security.DigestInputStream(java.nio.file.Files.newInputStream(path),digest)
    try stream.transferTo(java.io.OutputStream.nullOutputStream()) finally stream.close()
    digest.digest().map(b => f"${b & 255}%02x").mkString
  private def floats(bytes: Array[Byte]): Array[Float] =
    val buffer=java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    Array.fill(bytes.length/4)(buffer.getFloat())

  test("prepared CPU inference preserves batched tied models and generation") {
    for tied <- Seq(false,true) do
      val model=LlamaModel.init(config.copy(tieWordEmbeddings=tied),7)
      val prepared=BackendLlama.prepare(model,CpuFloatBackend)
      scala.util.Using.resource(prepared) { _ =>
        compare(prepared.logits(input).toFloatArray,model.logits(input).toFloatArray,1e-6)
        assertEquals(prepared.generateIds(Vector(1L,3L),3),model.generateIds(Vector(1L,3L),3,useCache=false))
        val policy=SamplingConfig(temperature=.7,topK=4,topP=.9,seed=42)
        assertEquals(prepared.generateIds(Vector(1L,3L),3,policy),model.generateIds(Vector(1L,3L),3,useCache=false,sampling=policy))
        val empty=Tensor.fromLongs(Array.emptyLongArray,Shape.from(Seq(2L,0L)))
        assertEquals(prepared.logits(empty).shape,Shape.from(Seq(2L,0L,8L)))
        model.norm.weight.load(model.norm.weight.data)
        intercept[IllegalArgumentException](prepared.logits(input))
      }
      prepared.close()
      intercept[IllegalStateException](prepared.logits(input))
  }

  test("CUDA prepared models retain only weights between forwards and clean failures") {
    gpu()
    val model=LlamaModel.init(config,7)
    var residentBytes=0L
    CudaSession.withDevice() { session =>
      val prepared=BackendLlama.prepare(model,session)
      val weights=session.allocatedBytes
      residentBytes=weights
      scala.util.Using.resource(prepared) { _ =>
        compare(prepared.logits(input).toFloatArray,model.logits(input).toFloatArray,3e-6)
        assertEquals(session.allocatedBytes,weights)
        intercept[IllegalArgumentException](prepared.logits(Tensor.fromLongs(Array(-1L),Shape.from(Seq(1L,1L)))))
        assertEquals(session.allocatedBytes,weights)
      }
      assertEquals(session.allocatedBytes,0L)
    }
    CudaSession.withDevice(maxBytes=residentBytes+64) { session =>
      val prepared=BackendLlama.prepare(model,session)
      scala.util.Using.resource(prepared) { _ =>
        intercept[IllegalArgumentException](prepared.logits(input))
        assertEquals(session.allocatedBytes,residentBytes)
      }
      assertEquals(session.allocatedBytes,0L)
    }
    CudaSession.withDevice(maxBytes=300) { session =>
      intercept[IllegalArgumentException](BackendLlama.prepare(model,session))
      assertEquals(session.allocatedBytes,0L)
    }
  }

  test("complete CUDA tiny Llama matches independent Candle full logits") {
    gpu()
    val dir=sys.props.get("scalatorch.llama.dir").getOrElse(assume(false,"public tiny model disabled").toString)
    val path=java.nio.file.Path.of(dir)
    assertEquals(sha(path.resolve("config.json")),"5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382")
    assertEquals(sha(path.resolve("model.safetensors")),"49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992")
    val model=LlamaModel.load(path)
    val tokens=Tensor.fromLongs(Array(1L,42L,17L,321L),Shape.from(Seq(1L,4L)))
    val expected=floats(resource("/llama/tiny-candle-f32.bin"))
    CudaSession.withDevice() { session =>
      val prepared=BackendLlama.prepare(model,session)
      scala.util.Using.resource(prepared) { _ =>
        val error=compare(prepared.logits(tokens).toFloatArray,expected,1e-5)
        println(s"CUDA tiny Llama Candle max absolute error: $error")
        assertEquals(prepared.generateIds(Vector(1L,42L,17L,321L),1).last,16196L)
      }
      assertEquals(session.allocatedBytes,0L)
    }
  }

  test("complete CUDA SmolLM2 matches Candle logits and trained greedy text") {
    gpu()
    val dir=sys.props.get("scalatorch.smollm.model.dir").getOrElse(assume(false,"trained model disabled").toString)
    val path=java.nio.file.Path.of(dir)
    assertEquals(sha(path.resolve("config.json")),"8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a")
    assertEquals(sha(path.resolve("model.safetensors")),"5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c")
    assertEquals(sha(path.resolve("tokenizer.json")),"9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c")
    val model=LlamaModel.load(path)
    val tokenizer=ByteBpeTokenizer.read(path.resolve("tokenizer.json"))
    val lines=new String(resource("/llama/smollm2/ids.txt"),java.nio.charset.StandardCharsets.UTF_8).linesIterator.toVector
    val prompt=lines(0).split(",").map(_.toLong).toVector
    val continuation=lines(1).split(",").map(_.toLong).toVector
    assertEquals(tokenizer.encode(SmolChat.format(Vector(ChatMessage("user","What is the capital of France?")))).toVector,prompt)
    val reference=floats(resource("/llama/smollm2/logits-f32.bin"))
    CudaSession.withDevice() { session =>
      val prepared=BackendLlama.prepare(model,session)
      val resident=session.allocatedBytes
      scala.util.Using.resource(prepared) { _ =>
        var maximum=0.0
        for step <- 0 until 4 do
          val ids=prompt++continuation.take(step)
          val result=prepared.logits(Tensor.fromLongs(ids.toArray,Shape.from(Seq(1L,ids.size.toLong)))).toFloatArray.takeRight(model.config.vocabSize)
          maximum=math.max(maximum,compare(result,reference.slice(step*model.config.vocabSize,(step+1)*model.config.vocabSize),2e-4))
          assertEquals(session.allocatedBytes,resident)
        println(s"CUDA SmolLM2 Candle max absolute error: $maximum residentBytes=$resident")
        Using.resource(prepared.newCache(capacity=prompt.size+12)) { cache =>
          val cachedBytes=session.allocatedBytes
          var cachedMaximum=0.0
          for step <- 0 until 4 do
            val chunk=if step==0 then prompt else Vector(continuation(step-1))
            val result=cache.decode(Tensor.fromLongs(chunk.toArray,Shape.from(Seq(1L,chunk.size.toLong)))).toFloatArray.takeRight(model.config.vocabSize)
            cachedMaximum=math.max(cachedMaximum,compare(result,reference.slice(step*model.config.vocabSize,(step+1)*model.config.vocabSize),2e-4))
            assertEquals(session.allocatedBytes,cachedBytes)
          println(s"CUDA cached SmolLM2 Candle max absolute error: $cachedMaximum cacheBytes=${cachedBytes-resident}")
        }
        assertEquals(session.allocatedBytes,resident)
        val generated=prepared.generateIds(prompt,12).drop(prompt.size)
        assertEquals(generated,continuation)
        assertEquals(tokenizer.decode(generated.toArray),new String(resource("/llama/smollm2/text.txt"),java.nio.charset.StandardCharsets.UTF_8))
      }
      assertEquals(session.allocatedBytes,0L)
    }
  }
