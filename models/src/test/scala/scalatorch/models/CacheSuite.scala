package scalatorch.models

import scalatorch.*
import scalatorch.nn.*
import java.nio.file.Path

class CacheSuite extends munit.FunSuite:
  private val config = LlamaConfig(8,8,16,2,4,2,maxPositions=16)
  private val ids = Tensor.fromLongs(Array(1L,2L,3L,4L,4L,3L,2L,1L), Shape.from(Seq(2L,4L)))
  private def near(a: Array[Float], b: Array[Float]): Unit =
    assertEquals(a.length,b.length)
    a.zip(b).foreach((x,y) => assertEqualsDouble(x,y,1e-5))
  test("batched GQA chunked decode matches all full-prefix positions") {
    val model = LlamaModel.init(config,7,trainable=false)
    val expected = model.logits(ids)
    for chunks <- Seq(Vector(1,1,1,1),Vector(2,2),Vector(1,3),Vector(4)) do
      val cache = model.newCache(batchSize=2,capacity=8)
      try
        var start=0
        chunks.foreach { count =>
          val result = cache.decode(ids.slice(1,start,start+count))
          near(result.toFloatArray,expected.slice(1,start,start+count).toFloatArray)
          start += count
          assertEquals(cache.position,start)
        }
      finally cache.close()
  }
  test("invalid calls preserve position and reset allows binding updated weights") {
    val model = LlamaModel.init(config,2,trainable=false)
    val cache = model.newCache(batchSize=2,capacity=4)
    try
      val first = cache.decode(ids.slice(1,0,2))
      val snapshot = first.toFloatArray
      val bad = Tensor.fromLongs(Array(1L,99L),Shape.from(Seq(2L,1L)))
      intercept[IllegalArgumentException](cache.decode(bad))
      intercept[IllegalArgumentException](cache.decode(ids))
      assertEquals(cache.position,2)
      near(cache.decode(ids.slice(1,2,3)).toFloatArray,model.logits(ids.slice(1,0,3)).slice(1,2,3).toFloatArray)
      val p = model.embedding.weight
      p.load(Tensor.fromFloats(p.data.toFloatArray.map(_+0.01f),p.data.shape))
      intercept[IllegalArgumentException](cache.decode(ids.slice(1,3,4)))
      assertEquals(cache.position,3)
      cache.reset()
      assertEquals(cache.position,0)
      near(cache.decode(ids).toFloatArray,model.logits(ids).toFloatArray)
      near(first.toFloatArray,snapshot)
    finally cache.close()
  }
  test("cache budgets close and late failures do not publish partial state") {
    val model = LlamaModel.init(config,1)
    intercept[IllegalArgumentException](model.newCache(batchSize=2,capacity=8,maxBytes=1))
    intercept[IllegalArgumentException](model.newCache(capacity=17))
    intercept[IllegalArgumentException](model.newCache(batchSize=Int.MaxValue,capacity=16,maxBytes=Long.MaxValue))
    val cache = model.newCache(batchSize=2,capacity=8)
    val expectedBytes = 2L*2*2*2*8*2*4
    assertEquals(cache.allocatedBytes,expectedBytes)
    val output = cache.decode(ids)
    cache.close(); cache.close()
    assertEquals(cache.allocatedBytes,0L)
    intercept[IllegalStateException](cache.decode(ids))
    intercept[IllegalStateException](cache.reset())
    assert(output.toFloatArray.forall(_.isFinite))
    val wrongHead = Linear(Parameter(Tensor.fromFloats(new Array[Float](56),Shape.from(Seq(8L,7L)))),None)
    val broken = model.copy(lmHead=wrongHead).newCache(batchSize=2,capacity=8)
    try
      intercept[IllegalArgumentException](broken.decode(ids))
      assertEquals(broken.position,0)
    finally broken.close()
  }
  test("greedy cached generation agrees with full-prefix recomputation") {
    val model = LlamaModel.init(config.copy(eosTokenIds=Vector.empty),5,trainable=false)
    val prompt = Vector(1L,3L,2L)
    assertEquals(model.generateIds(prompt,5,useCache=true),model.generateIds(prompt,5,useCache=false))
  }
  test("published model cached chunks match independent Candle full-prefix logits") {
    val directory = sys.props.get("scalatorch.llama.dir")
    assume(directory.nonEmpty,"public model cache gate requires scalatorch.llama.dir")
    val model = LlamaModel.load(Path.of(directory.get))
    val input = Tensor.fromLongs(Array(1L,42L,17L,321L),Shape.from(Seq(1L,4L)))
    val stream = getClass.getResourceAsStream("/llama/tiny-candle-f32.bin")
    val bytes = try stream.readAllBytes() finally stream.close()
    val buffer = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    val reference = Array.fill(bytes.length/4)(buffer.getFloat())
    for chunks <- Seq(Vector(1,1,1,1),Vector(2,2),Vector(1,3)) do
      val cache = model.newCache(capacity=8)
      try
        var start=0
        chunks.foreach { count =>
          val actual=cache.decode(input.slice(1,start,start+count)).toFloatArray
          near(actual,reference.slice(start*32000,(start+count)*32000))
          start += count
        }
      finally cache.close()
    assertEquals(model.generateIds(Vector(1L,42L,17L,321L),4,true),model.generateIds(Vector(1L,42L,17L,321L),4,false))
  }
  test("chunk output limits do not cap total cached prefix length") {
    val model = LlamaModel.init(config,6,trainable=false,limits=LlamaLimits(maxLogitsElements=16))
    val cache = model.newCache(capacity=8)
    val input = Tensor.fromLongs(Array(1L,2L,3L,4L),Shape.from(Seq(1L,4L)))
    try
      intercept[IllegalArgumentException](model.logits(input))
      cache.decode(input.slice(1,0,2))
      intercept[IllegalArgumentException](cache.decode(input.slice(1,1,4)))
      assertEquals(cache.position,2)
      val last = cache.decode(input.slice(1,2,4))
      val reference = model.copy(limits=LlamaLimits()).logits(input).slice(1,2,4)
      near(last.toFloatArray,reference.toFloatArray)
      assertEquals(cache.position,4)
    finally cache.close()
  }
