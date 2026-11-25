package scalatorch.models

/** Reference timestep lists: NVIDIA Align Your Steps quickstart, SD 1.5 and SDXL. */
class PaperAysScheduleSuite extends munit.FunSuite:
  test("paper schedules retain the ten evaluation times and nonzero final noise level") {
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val fixtures=Vector(
      (false,Vector(999,850,736,645,545,455,343,233,124,24,0)),
      (true,Vector(999,845,730,587,443,310,193,116,53,13,0)))
    fixtures.foreach { (xl,indices) =>
      val schedule=DpmSchedule.paperAys(noise,xl)
      assertEquals(schedule.timesteps,indices.dropRight(1).map(_.toFloat))
      schedule.sigmas.zip(indices).foreach { (sigma,index) =>
        assertEqualsDouble(sigma,noise.sigmas(index),1e-12)
      }
      assert(schedule.sigmas.last>0.0,"paper schedule ends at training timestep zero, not sigma zero")
    }
    intercept[IllegalArgumentException] {
      DpmSchedule.paperAys(DiffusionNoiseSchedule.linear(20,.00085,.012,scaled=true),false)
    }
    intercept[IllegalArgumentException] {
      DpmSchedule.paperAys(DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=false),false)
    }
    intercept[IllegalArgumentException] {
      DpmSchedule.paperAys(DiffusionNoiseSchedule.linear(1000,.001,.02,scaled=true),true)
    }
    intercept[IllegalArgumentException](DpmSchedule.paperAys(noise,false,0))
    for steps <- Seq(1,5,15,20) do
      for xl <- Seq(false,true) do
        val s=DpmSchedule.paperAys(noise,xl,steps)
        assertEquals(s.timesteps.size,steps)
        assertEquals(s.sigmas.size,steps+1)
        assert(s.sigmas.last>0.0)
        assertEqualsDouble(s.sigmas.head,noise.sigmas(999),1e-12)
        assertEqualsDouble(s.sigmas.last,noise.sigmas(0),1e-12)
        assert(s.timesteps.forall(t => t>=0f && t<=999f))
  }

  test("SDE DPM++ follows Diffusers on all paper AYS steps with explicit noise") {
    import scalatorch.*
    import scalatorch.autograd.Variable
    import scalatorch.models.worker.NativeJson
    import scalatorch.models.worker.NativeJson.*
    val stream=this.getClass.getResourceAsStream("/ays-sde-reference.json")
    val fixture=try scalatorch.models.worker.NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8))
      .asInstanceOf[Obj].fields finally stream.close()
    val cases=fixture("cases").asInstanceOf[Obj].fields
    def floats(value: NativeJson): Array[Float]=value.asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
    def variable(values: Array[Float])=Variable(Tensor.fromFloats(values,Shape.from(Seq(2))),false)
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for (key,xl) <- Vector("sd15"->false,"sdxl"->true) do
      val data=cases(key).asInstanceOf[Obj].fields
      val schedule=DpmSchedule.paperAys(noise,xl)
      val expectedSigmas=floats(data("sigmas"))
      schedule.sigmas.zip(expectedSigmas).foreach((a,b)=>assertEqualsDouble(a,b,0.00005))
      val solver=DpmPlusPlusMultistepSde(schedule.sigmas)
      var sample=variable(floats(data("initial")))
      val rows=data("rows").asInstanceOf[Arr].values
      assertEquals(rows.size,10)
      for (row,i) <- rows.zipWithIndex do
        val fields=row.asInstanceOf[Obj].fields
        sample=solver.step(i,variable(floats(fields("prediction"))),sample,variable(floats(fields("noise")))).previous
        sample.data.toFloatArray.zip(floats(fields("output"))).foreach((a,b)=>assertEqualsFloat(a,b,0.0001f,s"$key step $i"))
  }

  test("tiny U-Net with paper AYS follows Diffusers SDE output") {
    import java.nio.file.{Files,Path}
    import scalatorch.models.worker.NativeJson.*
    val root=Path.of(sys.env.getOrElse("SCALATORCH_POSTDIFF_TINY", ".superpowers/models/tiny-sd-pipe")).toAbsolutePath
    assume(Files.exists(root.resolve("unet/config.json")),"tiny SD fixture required")
    val pipe=new EulerImagePipeline(ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")),
      PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=1f,torchCpuNoise=true)
    val schedule=DpmSchedule.paperAys(DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true),false)
    var count=0
    val actual=pipe.generateDpmMultistepSde(request,schedule,returnLatents=true,onProgress=_=>count+=1).toFloatArray
    val stream=this.getClass.getResourceAsStream("/ays-sde-tiny-reference.json")
    val data=try scalatorch.models.worker.NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8))
      .asInstanceOf[Obj].fields finally stream.close()
    val expected=data("latent").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
    assertEquals(count,10)
    assertEquals(actual.length,expected.length)
    val maxError=actual.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
    println(s"tiny SD SDE AYS max latent error=$maxError")
    assert(maxError<0.001,s"tiny SD SDE AYS max latent error=$maxError")
  }

  test("DpmSchedule.paperAysZeroTerminal matches paperAys timesteps and ends at sigma 0") {
    val noise=DiffusionNoiseSchedule.linear(1000,0.00085,0.012,scaled=true)
    val s=DpmSchedule.paperAysZeroTerminal(noise,sdxl=true,steps=10)
    assertEquals(s.timesteps.size,10)
    assertEquals(s.sigmas.size,11)
    assertEquals(s.sigmas.last,0.0)
  }
