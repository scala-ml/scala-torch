package scalatorch.models

import java.nio.file.{Files,Path}

/** Tiny CPU demo with an optional explicit Euler config; no implicit conversion from other samplers. */
object TinyDiffusionRun:
  def main(args: Array[String]): Unit =
    require(args.length>=3 && args.length<=6,
      "usage: TinyDiffusionRun <tiny-sd-root> <prompt> <new-output.png> [seed [euler-config.json [steps]]]")
    val root=Path.of(args(0));val output=Path.of(args(2)).toAbsolutePath
    val seed=if args.length>=4 then args(3).toLong else 42L
    val steps=if args.length==6 then args(5).toInt else 4
    // Validate scheduler settings before loading model weights.
    val scheduler=if args.length>=5 then EulerConfig.read(Path.of(args(4))).scheduler(steps)
      else EulerScheduler(DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).euler(steps))
    val tokenizer=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
    val pipeline=new EulerImagePipeline(tokenizer,PretrainedClip.load(root.resolve("text_encoder")),
      PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val image=pipeline.generate(DiffusionImageRequest(args(1),"",32,32,seed),scheduler.schedule,
      onProgress=p => println(s"Euler step ${p.completedSteps}/${p.totalSteps}, timestep=${p.timestep}"),
      prediction=scheduler.prediction)
    Files.createDirectories(output.getParent)
    DiffusionImageIO.writePng(image,output)
    println(s"Wrote CPU Float32 tiny-model image: $output (seed=$seed, JVM Gaussian sequence)")
