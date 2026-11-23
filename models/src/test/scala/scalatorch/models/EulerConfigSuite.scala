package scalatorch.models

class EulerConfigSuite extends munit.FunSuite:
  test("UTF-8 config files load and malformed or oversized files reject") {
    val path=java.nio.file.Files.createTempFile("euler-config", ".json")
    try
      java.nio.file.Files.writeString(path,"{}")
      assertEquals(EulerConfig.read(path).scheduler(4).prediction,DiffusionPrediction.Epsilon)
      java.nio.file.Files.write(path,Array[Byte](0xc3.toByte,0x28.toByte))
      intercept[java.nio.charset.CharacterCodingException](EulerConfig.read(path))
      java.nio.file.Files.write(path,new Array[Byte](1048577))
      intercept[IllegalArgumentException](EulerConfig.read(path))
    finally java.nio.file.Files.delete(path)
  }
  test("checkpoint coefficients prediction and spacing produce an explicit scheduler") {
    val c=EulerConfig.fromJson("""{"_class_name":"EulerDiscreteScheduler","num_train_timesteps":10,
      "beta_start":0.001,"beta_end":0.02,"beta_schedule":"scaled_linear",
      "prediction_type":"v_prediction","timestep_spacing":"leading","steps_offset":1}""")
    val scheduler=c.scheduler(4)
    assertEquals(scheduler.prediction,DiffusionPrediction.VPrediction)
    assertEquals(scheduler.schedule.timesteps,Vector(7f,5f,3f,1f))
    assertEquals(scheduler.schedule,DiffusionNoiseSchedule.linear(10,.001,.02,true)
      .euler(4,DiffusionTimestepSpacing.Leading,1))
  }
  test("explicit trained betas override generated coefficients") {
    val c=EulerConfig.fromJson("""{"num_train_timesteps":2,"trained_betas":[0.2,0.25]}""")
    assertEquals(c.scheduler(2).schedule,DiffusionNoiseSchedule.fromBetas(Vector(.2,.25)).euler(2))
  }
  test("Karras config selects sigma interpolation while retaining leading noise scaling") {
    val c=EulerConfig.fromJson("""{"use_karras_sigmas":true,"timestep_spacing":"leading"}""")
    val expected=c.noise.karrasEuler(4)
    assertEquals(c.scheduler(4).schedule,expected.copy(initialNoiseSigma=math.hypot(expected.initialNoiseSigma,1)))
  }
  test("PNDM model scheduler metadata supplies shared beta configuration to the selected native sampler") {
    val c=EulerConfig.fromJson("""{"_class_name":"PNDMScheduler","_diffusers_version":"0.6.0",
      "beta_end":0.012,"beta_schedule":"scaled_linear","beta_start":0.00085,
      "num_train_timesteps":1000,"set_alpha_to_one":false,"skip_prk_steps":true,
      "steps_offset":1,"trained_betas":null,"clip_sample":false}""")
    assertEquals(c.scheduler(4).schedule,
      DiffusionNoiseSchedule.linear(1000,0.00085,0.012,true).euler(4,DiffusionTimestepSpacing.Linspace,1))
  }
  test("official SDXL Euler config accepts inherited PNDM metadata") {
    val c=EulerConfig.fromJson("""{"_class_name":"EulerDiscreteScheduler",
      "beta_end":0.012,"beta_schedule":"scaled_linear","beta_start":0.00085,
      "clip_sample":false,"sample_max_value":1.0,"set_alpha_to_one":false,
      "skip_prk_steps":true,"timestep_spacing":"leading","steps_offset":1}""")
    assertEquals(c.scheduler(4).schedule,
      DiffusionNoiseSchedule.linear(1000,0.00085,0.012,true).euler(4,DiffusionTimestepSpacing.Leading,1))
    intercept[IllegalArgumentException](EulerConfig.fromJson("""{"clip_sample":"false"}"""))
  }
  test("unsupported semantics and malformed JSON fail instead of silently using defaults") {
    for json <- Vector("""{"use_exponential_sigmas":true}""", """{"prediction_type":"unknown"}""",
      """{"num_train_timesteps":2.5}""",
      """{"trained_betas":[0,0.2]}""", """{"beta_start":0.1,"beta_start":0.2}""",
      """{"unknown_option":false}""", "{} {}") do
      val failed=try { EulerConfig.fromJson(json);false } catch case _: IllegalArgumentException => true
      assert(failed,s"expected failure for $json")
  }
