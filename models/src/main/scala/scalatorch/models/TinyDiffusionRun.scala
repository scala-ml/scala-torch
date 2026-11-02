package scalatorch.models

import java.nio.file.{Files,Path}

/** Standalone tiny-sd-pipe execution demo; fixed explicit Euler profile, not a general pipeline-config loader. */
object TinyDiffusionRun:
  def main(args: Array[String]): Unit =
    require(args.length==3 || args.length==4,"usage: TinyDiffusionRun <tiny-sd-root> <prompt> <new-output.png> [seed]")
    val root=Path.of(args(0));val output=Path.of(args(2)).toAbsolutePath
    val seed=if args.length==4 then args(3).toLong else 42L
    val tokenizer=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
    val pipeline=new EulerImagePipeline(tokenizer,PretrainedClip.load(root.resolve("text_encoder")),
      PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).euler(4)
    val image=pipeline.generate(DiffusionImageRequest(args(1),"",32,32,seed),schedule,
      onProgress=p => println(s"Euler step ${p.completedSteps}/${p.totalSteps}, timestep=${p.timestep}"))
    Files.createDirectories(output.getParent)
    DiffusionImageIO.writePng(image,output)
    println(s"Wrote CPU Float32 tiny-model image: $output (seed=$seed, JVM Gaussian sequence)")
