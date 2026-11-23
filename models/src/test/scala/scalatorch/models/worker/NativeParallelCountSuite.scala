package scalatorch.models.worker

import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger

class NativeParallelCountSuite extends munit.FunSuite:
  import NativeJson.*

  private def context(events: ConcurrentLinkedQueue[(String,Map[String,NativeJson])]=new ConcurrentLinkedQueue())=
    new NativeWorkContext(() => false,() => true,(n,f) => { events.add((n,f));() })

  private def request(params: Map[String,NativeJson],outer: Map[String,NativeJson]=Map.empty)=
    NativeRequest("r","txt2img",Map("params"->Obj(params),"output_dir"->Str("out"),"hardware_info"->Obj(Map("enabled"->Bool(false))))++outer,None)

  private def base(n: Int,b: Int)=Map[String,NativeJson]("prompt"->Str("a cat"),"seed"->Number(100),"n_iter"->Number(n),"batch_size"->Number(b),
    "controlnet"->Arr(Vector.empty),"img2img"->Null,"hooks"->Obj(Map("model_keyword"->Obj(Map("order"->Str("x"))))))

  private def fakeResult(r: NativeRequest)=
    val g=r.params("params").asInstanceOf[Obj].fields
    val seed=g("seed").asInstanceOf[Number].value
    val b=g("batch_size").asInstanceOf[Number].value.toInt
    Map[String,NativeJson]("images"->Arr((0 until b).map(i => Obj(Map("seed"->Number(seed+i)))).toVector),"warnings"->Arr(Vector(Str("w"))),
      "timing_s"->Number(1),"model_name"->Str("m"))

  test("count groups run as separate requests with shifted seeds; images come back in index order") {
    val seen=new ConcurrentLinkedQueue[Map[String,NativeJson]]()
    val result=NativeParallelCount.run(request(base(3,2)),context(),usesCuda=true,keywordInert=(_,_,_) => true) { (r,c) =>
      seen.add(r.params("params").asInstanceOf[Obj].fields); fakeResult(r) }
    val images=result.get("images").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields("seed").asInstanceOf[Number].value.toInt)
    assertEquals(images,Vector(100,101,102,103,104,105))
    assertEquals(result.get("warnings"),Arr(Vector(Str("w"))))
    assert(seen.toArray.forall(g => g.asInstanceOf[Map[String,NativeJson]]("n_iter")==Number(1)))
    assertEquals(result.get("model_name"),Str("m"))
  }

  test("progress events are renumbered to the whole job") {
    val events=new ConcurrentLinkedQueue[(String,Map[String,NativeJson])]()
    NativeParallelCount.run(request(base(2,3)),context(events),usesCuda=true,keywordInert=(_,_,_) => true) { (r,c) =>
      c.event("progress",Map("image"->Number(1),"images"->Number(3)))
      fakeResult(r) }
    val got=events.toArray.map(_.asInstanceOf[(String,Map[String,NativeJson])]._2).map(f => (f("image"),f("images"))).toSet
    assertEquals(got,Set[(NativeJson,NativeJson)]((Number(1),Number(6)),(Number(4),Number(6))))
  }

  test("requests that use LoRA, hooks, tags, other keys or a single count are not split") {
    def split(r: NativeRequest,inert: Boolean=true)=NativeParallelCount.run(r,context(),usesCuda=true,keywordInert=(_,_,_) => inert)((r,_) => fakeResult(r))
    assert(split(request(base(2,1))).nonEmpty)
    assert(split(request(base(1,4))).isEmpty,"count 1 stays sequential")
    assert(split(request(base(2,1)),inert=false).isEmpty,"active model keywords change prompts per image")
    assert(split(request(base(2,1)+("loras"->Arr(Vector(Str("x")))))).isEmpty)
    assert(split(request(base(2,1)+("prompt"->Str("a <lora:x:1> cat")))).isEmpty)
    assert(split(request(base(2,1)+("hooks"->Obj(Map("deepcache"->Obj(Map("enabled"->Bool(true))))))) ).isEmpty)
    assert(split(request(base(2,1)+("some_new_option"->Str("on")))).isEmpty)
    assert(split(request(base(2,1),Map("manual_loras"->Arr(Vector(Str("x")))))).isEmpty)
    assert(NativeParallelCount.run(request(base(2,1)),context(),usesCuda=false,keywordInert=(_,_,_) => true)((r,_) => fakeResult(r)).isEmpty,"CPU runs stay sequential")
  }

  test("a group that fails with a memory error is repeated alone; other errors stop the job") {
    val attempts=new AtomicInteger()
    val first=new java.util.concurrent.atomic.AtomicBoolean(true)
    val ok=NativeParallelCount.run(request(base(2,1)),context(),usesCuda=true,keywordInert=(_,_,_) => true) { (r,c) =>
      attempts.incrementAndGet()
      if r.params("params").asInstanceOf[Obj].fields("seed")==Number(101) && first.getAndSet(false) then throw new IllegalArgumentException("CUDA session allocation budget exceeded")
      fakeResult(r) }
    assertEquals(ok.get("images").asInstanceOf[Arr].values.size,2)
    assertEquals(attempts.get(),3)
    intercept[IllegalStateException] {
      NativeParallelCount.run(request(base(2,1)),context(),usesCuda=true,keywordInert=(_,_,_) => true) { (r,c) => throw new IllegalStateException("boom") }
    }
  }
