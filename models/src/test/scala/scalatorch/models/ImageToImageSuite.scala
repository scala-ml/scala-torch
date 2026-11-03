package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import java.nio.file.Path

class ImageToImageSuite extends munit.FunSuite:
  test("nine-channel U-Net input order is sample then mask then masked-image latent") {
    val sample=Tensor.fromFloats(Array(0f,1f,2f,3f),Shape.from(Seq(1L,4L,1L,1L)))
    val mask=Tensor.fromFloats(Array(4f),Shape.from(Seq(1L,1L,1L,1L)))
    val masked=Tensor.fromFloats(Array(5f,6f,7f,8f),sample.shape)
    assertEquals(EulerImagePipeline.concatenateInpaintChannels(sample,mask,masked).toFloatArray.toVector,
      Vector(0f,1f,2f,3f,4f,5f,6f,7f,8f))
  }

  test("nine-channel inpainting concatenates mask and masked-image latents for prediction") {
    val dir=sys.props.get("scalatorch.vae.dir")
    assume(dir.nonEmpty,"public tiny model required")
    val root=Path.of(dir.get).getParent
    val vae=PretrainedVae.load(root.resolve("vae"))
    val base=PretrainedUNet.load(root.resolve("unet"))
    val clip=PretrainedClip.load(root.resolve("text_encoder"))
    val tokenizer=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
    val old=base.convIn.weight.data
    val (out,in,height,width)=old.shape.dims match
      case Vector(o,i,h,w) => (o.toInt,i.toInt,h.toInt,w.toInt)
      case other => fail(s"unexpected conv_in shape: $other")
    assertEquals(in,4)
    val sourceWeights=old.toFloatArray
    val widened=Array.fill(out*9*height*width)(0f)
    for o <- 0 until out; c <- 0 until 9; y <- 0 until height; x <- 0 until width do
      val index=((o*9+c)*height+y)*width+x
      if c<in then widened(index)=sourceWeights(((o*in+c)*height+y)*width+x)
      else widened(index)=0.025f*(c-in+1)
    val widenedWeight=scalatorch.nn.Parameter(Tensor.fromFloats(widened,Shape.from(Seq(out,9,height,width))))
    val nineUnet=base.copy(convIn=scalatorch.nn.Conv2d(widenedWeight,base.convIn.bias,base.convIn.config))
    val pipeline=new EulerImagePipeline(tokenizer,clip,nineUnet,vae)
    val request=DiffusionImageRequest("a red cat","blurry",8,8,42,guidanceScale=7.5f)
    val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).euler(2)
    val noise=Tensor.fromFloats(Array.tabulate(64)(i => math.cos(i*.2).toFloat),Shape.from(Seq(1L,4L,4L,4L)))
    val posterior=Tensor.fromFloats(Array.fill(64)(0f),noise.shape)
    val image=Tensor.fromFloats(Array.tabulate(192)(i => math.sin(i*.1).toFloat),Shape.from(Seq(1L,3L,8L,8L)))
    val mask=Tensor.fromFloats(Array.tabulate(64)(i => if i%8>=4 then 1f else 0f),Shape.from(Seq(1L,1L,8L,8L)))
    val otherMask=Tensor.fromFloats(Array.tabulate(64)(i => if i%8<4 then 1f else 0f),mask.shape)
    val maskedImage=Tensor.fromFloats(new Array[Float](192),image.shape)
    val source=DiffusionImageSource(image,.5,Some(posterior),Some(mask),Some(image),maskedImage=Some(maskedImage),finalOverlay=false)
    val first=pipeline.generate(request,schedule,Some(noise),source=Some(source))
    val changedMask=pipeline.generate(request,schedule,Some(noise),source=Some(source.copy(mask=Some(otherMask))))
    val changedImage=pipeline.generate(request,schedule,Some(noise),source=Some(source.copy(
      maskedImage=Some(image))))
    assert(first.toFloatArray.forall(_.isFinite))
    assert(first.toFloatArray.toVector!=changedMask.toFloatArray.toVector,
      "the nine-channel U-Net must receive the repaint mask")
    assert(first.toFloatArray.toVector!=changedImage.toFloatArray.toVector,
      "the nine-channel U-Net must receive masked-image latents")
    val textOnly=pipeline.generate(request,schedule,Some(noise))
    assertEquals(textOnly.shape,Shape.from(Seq(1L,3L,8L,8L)))
    if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
      val gpu=new EulerImagePipeline(tokenizer,clip,nineUnet,vae,
        vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
      val gpuOutput=gpu.generate(request,schedule,Some(noise),source=Some(source))
      gpuOutput.toFloatArray.zip(first.toFloatArray).foreach((a,b)=>assertEqualsDouble(a,b,2e-5))
    intercept[IllegalArgumentException](pipeline.generate(request,schedule,Some(noise),source=Some(
      source.copy(maskedImage=None))))
  }

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
    val mask=Tensor.fromFloats(Array.tabulate(64)(i => if i%8<4 then 0f else 1f),Shape.from(Seq(1L,1L,8L,8L)))
    val masked=DiffusionImageSource(input,.5,Some(posterior),Some(mask),Some(input))
    for dpm <- Seq(false,true) do
      val layout=DpmSchedule.karras(schedule,2)
      val maskedOutput=if dpm then pipeline.generateDpm(request,layout,Some(noise),source=Some(masked))
        else pipeline.generate(request,schedule.euler(2),Some(noise),source=Some(masked))
      val maskedReplay=if dpm then pipeline.generateDpm(request,layout,Some(noise),source=Some(masked))
        else pipeline.generate(request,schedule.euler(2),Some(noise),source=Some(masked))
      val actual=maskedOutput.toFloatArray
      val original=input.toFloatArray
      for c <- 0 until 3; y <- 0 until 8; x <- 0 until 4 do
        val i=c*64+y*8+x
        assertEquals(actual(i),original(i))
      assert((0 until 3).exists(c => (0 until 8).exists(y => (4 until 8).exists(x =>
        actual(c*64+y*8+x)!=original(c*64+y*8+x)))))
      assertEquals(maskedOutput.toFloatArray.toVector,maskedReplay.toFloatArray.toVector)
      val rawOutput=if dpm then pipeline.generateDpm(request,layout,Some(noise),source=Some(masked.copy(finalOverlay=false)))
        else pipeline.generate(request,schedule.euler(2),Some(noise),source=Some(masked.copy(finalOverlay=false)))
      val raw=rawOutput.toFloatArray
      assert((0 until 3).exists(c => (0 until 8).exists(y => (0 until 4).exists(x =>
        raw(c*64+y*8+x)!=original(c*64+y*8+x)))),"disabling final pixel overlay must leave decoded crop pixels intact")
      if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
        val gpu=new EulerImagePipeline(tokenizer,clip,unet,vae,
          vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
        val cudaOutput=if dpm then gpu.generateDpm(request,layout,Some(noise),source=Some(masked))
          else gpu.generate(request,schedule.euler(2),Some(noise),source=Some(masked))
        cudaOutput.toFloatArray.zip(actual).foreach((a,b) => assertEqualsDouble(a,b,2e-5))
    intercept[IllegalArgumentException](DiffusionImageSource(input,.5,Some(posterior),Some(mask),None))
    intercept[IllegalArgumentException](DiffusionImageSource(input,.5,Some(posterior),None,Some(input)))
    intercept[IllegalArgumentException](pipeline.generate(request,schedule.euler(2),source=Some(
      DiffusionImageSource(input,.5,Some(posterior),Some(Tensor.fromFloats(new Array[Float](16),Shape.from(Seq(1L,1L,4L,4L)))),Some(input)))))
    intercept[IllegalArgumentException](pipeline.generate(request,schedule.euler(2),source=Some(
      DiffusionImageSource(input,.5,Some(posterior),Some(Tensor.fromFloats(Array.fill(64)(Float.NaN),mask.shape)),Some(input)))))
    intercept[IllegalArgumentException] {
      pipeline.generate(request,schedule.euler(2),source=Some(
        DiffusionImageSource(input,.5,mask=Some(mask),originalImage=Some(input),maskedImage=Some(input))))
    }
  }
