package scalatorch.models

import java.nio.file.{Files,Path}
import scalatorch.{Shape,Tensor}

class CfgCacheSuite extends munit.FunSuite:
  test("CfgCacheConfig validates parameter bounds") {
    val d = CfgCacheConfig()
    assertEqualsDouble(d.start, 0.25, 1e-12)
    assertEquals(d.interval, 3)
    assertEqualsDouble(d.adaptiveThreshold, 0.0, 1e-12)
    assertEqualsDouble(d.hfWeight, 0.0, 1e-12)
    intercept[IllegalArgumentException](CfgCacheConfig(start = -0.1))
    intercept[IllegalArgumentException](CfgCacheConfig(start = 1.1))
    intercept[IllegalArgumentException](CfgCacheConfig(interval = 1))
    intercept[IllegalArgumentException](CfgCacheConfig(interval = 21))
    intercept[IllegalArgumentException](CfgCacheConfig(adaptiveThreshold = -0.01))
    intercept[IllegalArgumentException](CfgCacheConfig(adaptiveThreshold = 1.01))
    intercept[IllegalArgumentException](CfgCacheConfig(hfWeight = -0.05))
    intercept[IllegalArgumentException](CfgCacheConfig(hfWeight = 0.51))
  }

  test("shouldReuse obeys warmup, fixed interval, adaptive cosine trigger, and shape invalidation") {
    val shapeLow = Shape.from(Seq(1L, 4L, 4L, 4L))
    val shapeHigh = Shape.from(Seq(1L, 4L, 8L, 8L))
    def fill(shape: Shape, v: Float) = Tensor.fromFloats(Array.fill(shape.numel.toInt)(v), shape)

    // Warmup ceil(10 * 0.3) = 3 steps (0,1,2 full); interval = 3 -> 3,4 reuse, 5 full, 6,7 reuse, 8 full, 9 reuse
    val st = new CfgCacheState(CfgCacheConfig(start = 0.3, interval = 3), 10)
    assert(!st.shouldReuse(0, shapeLow))
    val pattern = (0 until 10).map { step =>
      val reuse = st.shouldReuse(step, shapeLow)
      if reuse then st.applyCached(step, fill(shapeLow, 2.0f))
      else st.record(step, fill(shapeLow, 1.0f), fill(shapeLow, 2.0f))
      reuse
    }
    assertEquals(pattern.toList, List(false, false, false, true, true, false, true, true, false, true))
    assertEquals(st.fullSteps, 5)
    assertEquals(st.cachedSteps, 5)
    // Resolution change invalidates cache immediately
    assert(!st.shouldReuse(6, shapeHigh))

    // Phase-locks to active DeepCache scheduled full steps (e.g. step 4 is full under interval=2, fullFirst=2 -> blocks CFG reuse even if interval=4)
    val stLock = new CfgCacheState(CfgCacheConfig(start = 0.2, interval = 4), 10)
    stLock.record(3, fill(shapeLow, 1.0f), fill(shapeLow, 2.0f))
    DeepCache.install(Some(DeepCacheConfig(interval = 2, branch = 0, fullFirst = 2)))
    try
      assert(!stLock.shouldReuse(4, shapeLow))
      assert(stLock.shouldReuse(5, shapeLow))
    finally DeepCache.install(None)

    // Adaptive Guidance: high cosine similarity triggers reuse immediately after warmup even past interval
    val stAg = new CfgCacheState(CfgCacheConfig(start = 0.2, interval = 2, adaptiveThreshold = 0.99), 10)
    stAg.record(0, fill(shapeLow, 1.0f), fill(shapeLow, 3.0f))
    stAg.record(1, fill(shapeLow, 1.0f), fill(shapeLow, 3.0f))
    assert(stAg.cosineSimilarity > 0.999)
    assert((2 until 10).forall(s => stAg.shouldReuse(s, shapeLow)))
  }

  test("FasterCache hfWeight preserves flat fields and amplifies high-frequency checkerboard late in sampling") {
    val shape = Shape.from(Seq(1L, 1L, 4L, 4L))
    val st = new CfgCacheState(CfgCacheConfig(start = 0.0, interval = 3, hfWeight = 0.2), 10)
    val neg = Tensor.fromFloats(Array.fill(16)(0f), shape)
    // Interior pixels of 4x4 checkerboard: difference pos-neg is +/-1
    val checker = Array.tabulate(16)(i => if ((i / 4 + i % 4) % 2) == 0 then 1f else -1f)
    val pos = Tensor.fromFloats(checker, shape)
    st.record(0, neg, pos)
    // Step 2 (< 0.5 progress): no HF boost
    val early = st.applyCached(2, pos).toFloatArray
    assert(early.forall(v => math.abs(v) < 1e-6f))
    // Step 8 (>= 0.5 progress): center checkerboard pixels amplified by 1 + 0.2 -> pos - 1.2 * checker = -0.2 * checker
    val late = st.applyCached(8, pos).toFloatArray
    val centerIdx = 1 * 4 + 1 // (1,1) -> +1
    assertEqualsDouble(late(centerIdx).toDouble, -0.2, 1e-5)

    // Constant delta is unaffected by high-freq boost
    val stFlat = new CfgCacheState(CfgCacheConfig(start = 0.0, interval = 3, hfWeight = 0.2), 10)
    stFlat.record(0, Tensor.fromFloats(Array.fill(16)(1f), shape), Tensor.fromFloats(Array.fill(16)(3f), shape))
    val flatOut = stFlat.applyCached(8, Tensor.fromFloats(Array.fill(16)(5f), shape)).toFloatArray
    assert(flatOut.forall(v => math.abs(v - 3f) < 1e-5))
  }

  test("EulerImagePipeline runs CFG-Cache deterministically and skips negative branch on CPU") {
    val root = Path.of(sys.env.getOrElse("SCALATORCH_POSTDIFF_TINY", ".superpowers/models/tiny-sd-pipe")).toAbsolutePath
    assume(Files.exists(root.resolve("unet/config.json")), "tiny SD fixture required")
    val pipe = new EulerImagePipeline(ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")),
      PretrainedClip.load(root.resolve("text_encoder")), PretrainedUNet.load(root.resolve("unet")), PretrainedVae.load(root.resolve("vae")))
    val schedule = EulerSchedule(Vector(800f, 600f, 400f, 200f, 0f), Vector(2.0, 1.5, 1.0, 0.5, 0.1, 0.0), 2.0)
    val base = DiffusionImageRequest("a red cat", "blurry", 16, 16, 42, guidanceScale = 7f)
    val withCache = base.copy(cfgCache = Some(CfgCacheConfig(start = 0.2, interval = 3)))
    val plainOut = pipe.generateAncestralLatents(base, schedule).toFloatArray
    val cached1 = pipe.generateAncestralLatents(withCache, schedule).toFloatArray
    val cached2 = pipe.generateAncestralLatents(withCache, schedule).toFloatArray
    assertEquals(cached1.toVector, cached2.toVector)
    assertNotEquals(cached1.toVector, plainOut.toVector)
  }

  test("deferUntil blocks recording and reuse until after PostDiff transition") {
    val shapeLow = Shape.from(Seq(1L, 4L, 4L, 4L))
    val shapeHigh = Shape.from(Seq(1L, 4L, 8L, 8L))
    def fill(shape: Shape, v: Float) = Tensor.fromFloats(Array.fill(shape.numel.toInt)(v), shape)
    val st = new CfgCacheState(CfgCacheConfig(start = 0.0, interval = 3), 20)
    st.deferUntil(10) // e.g. transitionIndex = 8 -> skip recording 0..8, record 9 at high-res, allow reuse from 10+
    for step <- 0 to 8 do
      assert(!st.shouldReuse(step, shapeLow))
      st.record(step, fill(shapeLow, 1.0f), fill(shapeLow, 2.0f))
    assertEquals(st.fullSteps, 0)
    assert(!st.shouldReuse(9, shapeHigh))
    st.record(9, fill(shapeHigh, 1.0f), fill(shapeHigh, 2.0f))
    assertEquals(st.fullSteps, 1)
    assert(st.shouldReuse(10, shapeHigh))
  }
