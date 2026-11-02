package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import java.nio.file.Path

class ImageToImageSuite extends munit.FunSuite:
  test("image initialization matches explicit Euler and VP DPM suffix sampling") {
    val dir=sys.props.get("scalatorch.vae.dir")
    assume(dir.nonEmpty,"public tiny model required")
    val root=Path.of(dir.get).getParent
    val vae=PretrainedVae.load(root.resolve("vae"))
    val unet=PretrainedUNet.load(root.resolve("unet"))
    val clip=PretrainedClip.load(root.resolve("text_encoder"))
    val tokenizer=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
    val pipeline=new EulerImagePipeline(tokenizer,clip,unet,vae)
    val request=DiffusionImageRequest("a cat","",8,8,42,guidanceScale=1)
    val input=Tensor.fromFloats(Array.tabulate(192)(i => math.sin(i*.1).toFloat),Shape.from(Seq(1L,3L,8L,8L)))
    val noise=Tensor.fromFloats(Array.tabulate(64)(i => math.cos(i*.2).toFloat),Shape.from(Seq(1L,4L,4L,4L)))
    val posterior=Tensor.fromFloats(Array.fill(64)(0f),noise.shape)
    val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for dpm <- Seq(false,true) do
      val full=if dpm then DpmSchedule.karras(schedule,4).layout else schedule.euler(4)
      val suffix=EulerSchedule(full.timesteps.drop(2),full.sigmas.drop(2),full.initialNoiseSigma)
      val expected=Autograd.noGrad {
        val clean=vae.toDiffusionLatents(vae.encode(Variable(input)).sample(posterior))
        val sigma=suffix.sigmas.head
        val alpha=if dpm then 1/math.hypot(1,sigma) else 1.0
        var x=clean*alpha.toFloat+Variable(noise)*(sigma*alpha).toFloat
        val context=clip(tokenizer.batch(Vector(request.prompt)).ids)
        val euler=EulerScheduler(suffix)
        val solver=DpmPlusPlus2M(suffix.sigmas)
        var history: Option[Variable]=None
        for i <- suffix.timesteps.indices do
          val prediction=unet(if dpm then x else euler.scaleModelInput(i,x),
            Tensor.fromFloats(Array(suffix.timesteps(i)),Shape.from(Seq(1L))),context)
          if dpm then
            val step=solver.step(i,prediction,x,history);x=step.previous;history=Some(step.predictedOriginal)
          else x=euler.step(i,prediction,x).previous
        vae.decodeDiffusionLatents(x).data
      }
      var steps=Vector.empty[Int]
      val source=DiffusionImageSource(input,.5,Some(posterior))
      val actual=if dpm then pipeline.generateDpm(request,DpmSchedule.karras(schedule,4),Some(noise),
        onProgress=p => steps :+= p.completedSteps,source=Some(source))
      else pipeline.generate(request,full,Some(noise),onProgress=p => steps :+= p.completedSteps,source=Some(source))
      assertEquals(steps,Vector(1,2))
      assertEquals(actual.toFloatArray.toVector,expected.toFloatArray.toVector)
      if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
        val gpu=new EulerImagePipeline(tokenizer,clip,unet,vae,
          vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
        val output=if dpm then gpu.generateDpm(request,DpmSchedule.karras(schedule,4),Some(noise),source=Some(source))
          else gpu.generate(request,full,Some(noise),source=Some(source))
        output.toFloatArray.zip(expected.toFloatArray).foreach((a,b) => assertEqualsDouble(a,b,2e-5))
    var count=0
    pipeline.generate(request,schedule.euler(4),Some(noise),onProgress=_ => count+=1,
      source=Some(DiffusionImageSource(input,0,Some(posterior))))
    assertEquals(count,1)
    intercept[IllegalArgumentException](pipeline.generate(request,schedule.euler(4),
      source=Some(DiffusionImageSource(input,Double.NaN))))
    intercept[IllegalArgumentException](pipeline.generate(request,schedule.euler(4),
      source=Some(DiffusionImageSource(input,1.1))))
    intercept[IllegalArgumentException](pipeline.generate(request,schedule.euler(4),
      source=Some(DiffusionImageSource(noise,.5))))
    val seeded=DiffusionImageSource(input,.5)
    val first=pipeline.generate(request,schedule.euler(2),source=Some(seeded))
    val replay=pipeline.generate(request,schedule.euler(2),source=Some(seeded))
    assertEquals(first.toFloatArray.toVector,replay.toFloatArray.toVector)
    intercept[java.util.concurrent.CancellationException](pipeline.generate(request,schedule.euler(2),
      source=Some(seeded),cancelled=() => true))
  }
