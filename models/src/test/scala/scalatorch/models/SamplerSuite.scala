package scalatorch.models

class SamplerSuite extends munit.FunSuite:
  private def near(actual: Array[Double], expected: Double*): Unit =
    assertEquals(actual.length,expected.size)
    actual.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,1e-7))
  test("stable temperature softmax and greedy tie semantics") {
    near(TokenSampler.probabilities(Array(0f,math.log(4).toFloat),SamplingConfig(temperature=2)),1.0/3,2.0/3)
    near(TokenSampler.probabilities(Array(2f,2f,1f),SamplingConfig(topK=1,topP=0.01)),1,0,0)
    near(TokenSampler.probabilities(Array(Float.MaxValue,-Float.MaxValue),SamplingConfig(temperature=Double.MinPositiveValue)),1,0)
    assert(TokenSampler.probabilities(Array(0f,-50f,-100f),SamplingConfig(temperature=1,topP=1)).forall(_ > 0))
    near(TokenSampler.probabilities(Array(1f,1f,1f),SamplingConfig(temperature=1,topK=99)),1.0/3,1.0/3,1.0/3)
  }
  test("signed zero logits use numeric equality for lowest-ID ties") {
    near(TokenSampler.probabilities(Array(-0.0f,0.0f),SamplingConfig(temperature=1,topK=1)),1,0)
    near(TokenSampler.probabilities(Array(-0.0f,0.0f),SamplingConfig(temperature=1,topP=0.5)),1,0)
  }
  test("top-k then nucleus keeps threshold crossing token and renormalizes") {
    val logits = Array(math.log(0.6).toFloat,math.log(0.3).toFloat,math.log(0.1).toFloat)
    near(TokenSampler.probabilities(logits,SamplingConfig(temperature=1,topP=0.7)),2.0/3,1.0/3,0)
    near(TokenSampler.probabilities(logits,SamplingConfig(temperature=1,topK=2,topP=0.65)),1,0,0)
    near(TokenSampler.probabilities(Array(0f,0f,0f,0f),SamplingConfig(temperature=1,topP=0.5)),0.5,0.5,0,0)
    near(TokenSampler.probabilities(Array(0f,0f,0f),SamplingConfig(temperature=1,topP=Double.MinPositiveValue)),1,0,0)
  }
  test("invalid configuration and empty or nonfinite scores reject") {
    for temperature <- Seq(-1.0,Double.NaN,Double.PositiveInfinity) do intercept[IllegalArgumentException](SamplingConfig(temperature=temperature))
    for p <- Seq(0.0,-0.1,1.01,Double.NaN) do intercept[IllegalArgumentException](SamplingConfig(topP=p))
    intercept[IllegalArgumentException](SamplingConfig(topK = -1))
    val sampler = new TokenSampler(SamplingConfig())
    for scores <- Seq(Array.emptyFloatArray,Array(Float.NaN),Array(Float.PositiveInfinity),Array(Float.NegativeInfinity)) do
      intercept[IllegalArgumentException](sampler.sample(scores))
  }
  test("seeded streams repeat and categorical frequencies agree with probabilities") {
    val scores = Array(math.log(0.25).toFloat,math.log(0.75).toFloat)
    val a = new TokenSampler(SamplingConfig(temperature=1,seed=42))
    val b = new TokenSampler(SamplingConfig(temperature=1,seed=42))
    val samples = Vector.fill(20000)(a.sample(scores))
    assertEquals(samples,Vector.fill(20000)(b.sample(scores)))
    assertEqualsDouble(samples.count(_ == 0L).toDouble/samples.size,0.25,0.015)
    assert(samples.forall(id => id == 0 || id == 1))
  }
  test("Llama cached and full-prefix generation share seeded policy and replay") {
    val model = LlamaModel.init(LlamaConfig(8,4,8,1,1,1,maxPositions=16,eosTokenIds=Vector.empty),3,trainable=false)
    val policy = SamplingConfig(temperature=0.8,topK=5,topP=0.9,seed=17)
    val actual = model.generateIds(Vector(1L,3L),6,sampling=policy)
    assertEquals(actual,model.generateIds(Vector(1L,3L),6,sampling=policy))
    assertEquals(actual,model.generateIds(Vector(1L,3L),6,useCache=false,sampling=policy))
    assertEquals(actual.size,8)
    assertEquals(model.generateIds(Vector(1L),2),model.generateIds(Vector(1L),2,sampling=SamplingConfig()))
  }
