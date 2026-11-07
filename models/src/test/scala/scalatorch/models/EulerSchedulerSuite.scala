package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable

class EulerSchedulerSuite extends munit.FunSuite:
  private def v(xs: Float*)=Variable(Tensor.fromFloats(xs.toArray,Shape.from(Seq(xs.size))))
  test("Heun averages current and predicted next derivatives and uses Euler at terminal sigma zero") {
    val schedule=EulerSchedule(Vector(2f,1f),Vector(2.0,1.0,0.0),2.0)
    val solver=HeunScheduler(schedule)
    val first=solver.predict(0,v(1f),v(10f))
    assertEqualsFloat(first.predictor.data.toFloatArray(0),9f,1e-6f,"")
    val corrected=solver.correct(0,first,v(2f))
    assertEqualsFloat(corrected.previous.data.toFloatArray(0),8.5f,1e-6f,"")
    val terminal=solver.predict(1,v(2f),corrected.previous)
    assert(!terminal.requiresSecondEvaluation,"")
    assertEqualsFloat(terminal.predictor.data.toFloatArray(0),6.5f,1e-6f,"")
  }
  test("Karras sigmas use rho interpolation and logarithmic model timesteps") {
    val noise=DiffusionNoiseSchedule.fromBetas(Vector(.2,.375,.6))
    val schedule=noise.karrasEuler(3,rho=1)
    schedule.sigmas.zip(Vector(2.0,1.25,.5,0.0)).foreach((a,b) => assertEqualsDouble(a,b,1e-14))
    assertEqualsFloat(schedule.timesteps(0),2f,1e-6f)
    assertEqualsFloat(schedule.timesteps(1),(1+math.log(1.25)/math.log(2)).toFloat,1e-6f)
    assertEqualsFloat(schedule.timesteps(2),0f,1e-6f)
    assertEqualsDouble(noise.karrasEuler(1).sigmas.head,2.0,1e-14)
    assertEqualsDouble(noise.karrasEuler(3,rho=2).sigmas(1),1.125,1e-14)
    assertEquals(DiffusionNoiseSchedule.fromBetas(Vector(.2)).karrasEuler(1).timesteps,Vector(0f))
    for rho <- Vector(0.0,-1.0,Double.NaN,Double.PositiveInfinity) do
      intercept[IllegalArgumentException](noise.karrasEuler(3,rho))
    intercept[IllegalArgumentException](noise.karrasEuler(0))
  }
  test("Align Your Steps matches the NUI Diffusers reference for SD1 and SDXL") {
    import scalatorch.models.worker.NativeJson.*
    val input=this.getClass.getResourceAsStream("/ays-schedule-reference.json")
    val fixture=try scalatorch.models.worker.NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8))
      .asInstanceOf[Obj].fields finally input.close()
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for xl<-Vector(false,true); steps<-Vector(1,4,11,20) do
      val ref=fixture(s"${if xl then "True" else "False"}:$steps").asInstanceOf[Obj].fields
      val expectedSigmas=ref("sigmas").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toDouble)
      val expectedTimes=ref("timesteps").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val actual=noise.alignYourStepsEuler(steps,xl)
      assertEquals(actual.sigmas.size,expectedSigmas.size)
      actual.sigmas.zip(expectedSigmas).foreach((a,b)=>assertEqualsDouble(a,b,0.00002))
      actual.timesteps.zip(expectedTimes).foreach((a,b)=>assertEqualsFloat(a,b,0.002f))
  }
  test("KL Optimal preserves Diffusers' nonzero terminal sigma") {
    import scalatorch.models.worker.NativeJson.*
    val input=this.getClass.getResourceAsStream("/kl-schedule-reference.json")
    val fixture=try scalatorch.models.worker.NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8))
      .asInstanceOf[Obj].fields finally input.close()
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for steps<-Vector(1,4,11,20) do
      val ref=fixture(s"False:$steps").asInstanceOf[Obj].fields
      val expectedSigmas=ref("sigmas").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toDouble)
      val expectedTimes=ref("timesteps").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val actual=noise.klOptimalEuler(steps)
      assert(actual.sigmas.last>0)
      assertEquals(actual.sigmas.size,expectedSigmas.size)
      actual.sigmas.zip(expectedSigmas).foreach((a,b)=>assertEqualsDouble(a,b,0.00002))
      actual.timesteps.zip(expectedTimes).foreach((a,b)=>assertEqualsFloat(a,b,0.002f))
  }
  test("NUI DDIM schedule matches Euler, ancestral and DPM++ leading spacing") {
    import scalatorch.models.worker.NativeJson.*
    val input=this.getClass.getResourceAsStream("/ddim-schedule-reference.json")
    val fixture=try scalatorch.models.worker.NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8))
      .asInstanceOf[Obj].fields finally input.close()
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for sampler<-Vector("Euler","Euler a","DPM++ 2M"); offset<-Vector(0,1); steps<-Vector(1,4,7,11,20) do
      val ref=fixture(s"$sampler:$offset:$steps").asInstanceOf[Obj].fields
      val expectedSigmas=ref("sigmas").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toDouble)
      val expectedTimes=ref("timesteps").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val (times,sigmas)=sampler match
        case "Euler" => val value=noise.ddimEuler(steps);(value.timesteps,value.sigmas)
        case "Euler a" => val value=noise.euler(steps,DiffusionTimestepSpacing.Leading,offset);(value.timesteps,value.sigmas)
        case _ => val value=DpmSchedule.leading(noise,steps,offset);(value.timesteps,value.sigmas)
      assertEquals(times.size,expectedTimes.size,s"$sampler $steps")
      assertEquals(sigmas.size,expectedSigmas.size,s"$sampler $steps")
      sigmas.zip(expectedSigmas).foreach((a,b)=>assertEqualsDouble(a,b,0.00002))
      times.zip(expectedTimes).foreach((a,b)=>assertEqualsFloat(a,b,0.002f))
    assertEquals(noise.ddimEuler(7).timesteps.size,8,"NUI DDIM can yield more steps than requested")
  }
  test("checkpoint steps offset applies only to leading spacing") {
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for spacing <- Vector(DiffusionTimestepSpacing.Linspace,DiffusionTimestepSpacing.Trailing) do
      assertEquals(noise.euler(4,spacing,1),noise.euler(4,spacing,0))
    assertEquals(noise.euler(4,DiffusionTimestepSpacing.Leading,1).timesteps,
      Vector(751f,501f,251f,1f))
  }
  test("beta products determine sigma and interpolation with explicit spacing") {
    val noise=DiffusionNoiseSchedule.fromBetas(Vector(.2,.25))
    assertEqualsDouble(noise.sigmas(0),.5,1e-14)
    assertEqualsDouble(noise.sigmas(1),math.sqrt(2.0/3),1e-14)
    val schedule=noise.euler(2)
    assertEquals(schedule.timesteps,Vector(1f,0f))
    assertEquals(schedule.sigmas,noise.sigmas.reverse :+ 0.0)
    val ten=DiffusionNoiseSchedule.linear(10,.001,.02)
    assertEquals(ten.euler(4).timesteps,Vector(9f,6f,3f,0f))
    assertEquals(ten.euler(4,DiffusionTimestepSpacing.Leading,1).timesteps,Vector(7f,5f,3f,1f))
    assertEquals(ten.euler(4,DiffusionTimestepSpacing.Trailing).timesteps,Vector(9f,7f,4f,1f))
    assertEquals(ten.euler(1).timesteps,Vector(0f))
    val fractional=ten.euler(3)
    assertEquals(fractional.timesteps,Vector(9f,4.5f,0f))
    assertEqualsDouble(fractional.sigmas(1),(ten.sigmas(4)+ten.sigmas(5))*.5,1e-14)
    val leading=ten.euler(4,DiffusionTimestepSpacing.Leading)
    assertEqualsDouble(leading.initialNoiseSigma,math.hypot(leading.sigmas.head,1),1e-14)
    assertEqualsDouble(DiffusionNoiseSchedule.linear(3,.01,.09,scaled=true).betas(1),.04,1e-14)
  }
  test("Euler prediction modes agree on an analytical denoising step") {
    val schedule=EulerSchedule(Vector(9f,4f),Vector(2.0,1.0,0.0),2.0)
    val sample=v(3f,-2f)
    val clean=v(1f,2f)
    val epsilon=v(1f,-2f)
    val velocity=v(((3.0/5-1)*math.sqrt(5)/2).toFloat,((-2.0/5-2)*math.sqrt(5)/2).toFloat)
    for (prediction,output) <- Vector((DiffusionPrediction.Epsilon,epsilon),(DiffusionPrediction.Sample,clean),(DiffusionPrediction.VPrediction,velocity)) do
      val scheduler=EulerScheduler(schedule,prediction)
      val result=scheduler.step(0,output,sample)
      result.predictedOriginal.data.toFloatArray.zip(Array(1f,2f)).foreach((a,b) => assertEqualsFloat(a,b,1e-6f))
      result.previous.data.toFloatArray.zip(Array(2f,0f)).foreach((a,b) => assertEqualsFloat(a,b,1e-6f))
      val scaled=scheduler.scaleModelInput(0,sample).data.toFloatArray
      assertEqualsFloat(scaled(0),(3/math.sqrt(5)).toFloat,1e-6f)
  }
  test("complete Euler trajectory recovers clean input and is stateless with gradients") {
    val scheduler=EulerScheduler(EulerSchedule(Vector(9f,4f,0f),Vector(3.0,1.0,.5,0.0),3.0))
    val clean=v(2f,-1f);val epsilon=v(.25f,-.5f)
    var sample=scheduler.addNoise(0,clean,epsilon)
    for i <- 0 until 3 do sample=scheduler.step(i,epsilon,sample).previous
    assertEquals(sample.data.toFloatArray.toVector,clean.data.toFloatArray.toVector)
    sample.sum().backward()
    assertEquals(clean.grad.get.toFloatArray.toVector,Vector(1f,1f))
    assertEquals(epsilon.grad.get.toFloatArray.toVector,Vector(0f,0f))
    assertEquals(scheduler.step(0,epsilon,v(3,4)).previous.data.toFloatArray.toVector,
      scheduler.step(0,epsilon,v(3,4)).previous.data.toFloatArray.toVector)
  }
  test("Euler rejects malformed schedules, shapes, step indices and allocation budgets") {
    for betas <- Vector(Vector.empty[Double],Vector(0.0),Vector(1.0),Vector(Double.NaN)) do
      intercept[IllegalArgumentException](DiffusionNoiseSchedule.fromBetas(betas))
    intercept[IllegalArgumentException](EulerSchedule(Vector(1f),Vector(0.0,0.0),1))
    intercept[IllegalArgumentException](EulerSchedule(Vector(1f,0f),Vector(1.0,2.0,0.0),2))
    intercept[IllegalArgumentException](EulerSchedule(Vector(1f),Vector(1.0,-0.1),1))
    intercept[IllegalArgumentException](EulerSchedule(Vector(1f),Vector(1.0,Double.NaN),1))
    intercept[IllegalArgumentException](EulerSchedule(Vector(1f),Vector(1.0,1.1),1))
    intercept[IllegalArgumentException](EulerSchedule(Vector(Float.NaN),Vector(1.0,0.0),1))
    val noise=DiffusionNoiseSchedule.linear(10,.001,.02)
    intercept[IllegalArgumentException](noise.euler(0))
    assertEquals(noise.euler(11).timesteps.size,11) // Linspace permits repeated/extra inference steps.
    intercept[IllegalArgumentException](noise.euler(10,DiffusionTimestepSpacing.Leading,1))
    val scheduler=EulerScheduler(noise.euler(2),maxElements=2)
    intercept[IllegalArgumentException](scheduler.step(2,v(1),v(1)))
    intercept[IllegalArgumentException](scheduler.step(0,v(1),v(1,2)))
    intercept[IllegalArgumentException](scheduler.scaleModelInput(0,v(1,2,3)))
    assertEquals(scheduler.step(0,v(),v()).previous.data.shape,Shape.from(Seq(0)))
  }
