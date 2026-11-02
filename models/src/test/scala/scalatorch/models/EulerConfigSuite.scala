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
  test("unsupported semantics and malformed JSON fail instead of silently using defaults") {
    for json <- Vector("""{"use_karras_sigmas":true}""", """{"prediction_type":"unknown"}""",
      """{"_class_name":"DDIMScheduler"}""", """{"num_train_timesteps":2.5}""",
      """{"trained_betas":[0,0.2]}""", """{"beta_start":0.1,"beta_start":0.2}""",
      """{"unknown_option":false}""", "{} {}", """{"rescale_betas_zero_snr":true}""") do
      intercept[IllegalArgumentException](EulerConfig.fromJson(json))
  }
