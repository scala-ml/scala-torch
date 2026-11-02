package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable

class DpmPlusPlusSuite extends munit.FunSuite:
  private def v(x: Float)=Variable(Tensor.fromFloats(Array(x),Shape.from(Seq(1))))
  test("DPM Karras schedules round model timesteps and use unit initial noise") {
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val schedule=DpmSchedule.karras(noise,4)
    assertEquals(schedule.timesteps,Vector(999f,687f,146f,0f))
    assertEquals(schedule.sigmas,noise.karrasEuler(4).sigmas)
    assertEquals(schedule.layout.initialNoiseSigma,1.0)
    intercept[IllegalArgumentException](DpmSchedule(Vector(2f,1f),Vector(2.0,2.0,0.0)))
  }
  test("VP midpoint uses history and terminal zero returns current clean prediction") {
    val solver=DpmPlusPlus2M(Vector(4.0,2.0,1.0,0.0),DiffusionPrediction.Sample)
    val first=solver.step(0,v(1),v(0))
    assertEqualsFloat(first.previous.data.toFloatArray.head,(.5/math.sqrt(5)).toFloat,1e-6f)
    val second=solver.step(1,v(3),first.previous,Some(first.predictedOriginal))
    assertEqualsFloat(second.previous.data.toFloatArray.head,(2.25/math.sqrt(2)).toFloat,1e-6f)
    assertEqualsFloat(solver.step(2,v(5),second.previous,Some(second.predictedOriginal)).previous.data.toFloatArray.head,5f,0f)
    intercept[IllegalArgumentException](solver.step(1,v(3),first.previous))
  }
  test("epsilon and velocity recover a known clean sample from VP input") {
    val alpha=1/math.sqrt(5);val sigma=2*alpha
    val clean=3.0;val noise=0.5;val sample=alpha*clean+sigma*noise
    for (mode,output) <- Vector((DiffusionPrediction.Epsilon,noise),
      (DiffusionPrediction.VPrediction,alpha*noise-sigma*clean)) do
      val result=DpmPlusPlus2M(Vector(2.0,0.0),mode).step(0,v(output.toFloat),v(sample.toFloat))
      assertEqualsFloat(result.previous.data.toFloatArray.head,clean.toFloat,1e-6f)
    intercept[IllegalArgumentException](DpmPlusPlus2M(Vector(2.0,2.0,0.0)))
  }
  test("history gradients and rejected updates preserve caller-owned state") {
    val solver=DpmPlusPlus2M(Vector(4.0,2.0,1.0,0.0),DiffusionPrediction.Sample)
    val current=v(3);val old=v(1);val input=v(2)
    val result=solver.step(1,current,input,Some(old))
    result.previous.sum().backward()
    assertEqualsFloat(current.grad.get.toFloatArray.head,(.75/math.sqrt(2)).toFloat,1e-6f)
    assertEqualsFloat(old.grad.get.toFloatArray.head,(-.25/math.sqrt(2)).toFloat,1e-6f)
    intercept[IllegalArgumentException](solver.step(0,current,input,Some(old)))
    intercept[IllegalArgumentException](solver.copy(maxElements=0).step(1,current,input,Some(old)))
    assertEquals(solver.step(1,current,input,Some(old)).previous.data.toFloatArray.toVector,
      result.previous.data.toFloatArray.toVector)
  }
