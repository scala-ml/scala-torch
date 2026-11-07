package scalatorch.models.worker

import java.nio.file.{Files,Path}

class NativeInpaintPipelineFamilySuite extends munit.FunSuite:
  test("loads the Stable Diffusion inpainting pipeline family") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val source=Path.of(configured.get).getParent
    val model=Files.createTempDirectory("native-inpaint-family")
    val scheduler=model.resolve("scheduler.json")
    try
      new NativeDiffusionBackendSuite().makeNineChannelModel(source,model)
      Files.writeString(model.resolve("model_index.json"),"""{"_class_name":"StableDiffusionInpaintPipeline"}""")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(model,scheduler)
      val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
      backend.execute(NativeRequest("load","load_model",Map("path"->NativeJson.Str(model.toString)),None),context)
      assert(backend.features.nonEmpty)
    finally
      val paths=Files.walk(model)
      try
        import scala.jdk.CollectionConverters.*
        paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally paths.close()
  }
