package scalatorch.models

import java.nio.file.{Files,Path}
import scalatorch.{Shape,Tensor}

/** Opt-in real-checkpoint verification for SD2 single-file loading against the released diffusers directory. */
class SingleFileSd2RealSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(10,"minutes")

  private def setting(name: String): Option[String] =
    val properties=new java.util.Properties()
    val config=Vector(Path.of(".superpowers/sd2-single-file-test.properties"),
      Path.of("../.superpowers/sd2-single-file-test.properties")).find(Files.isRegularFile(_))
    config.foreach(path => scala.util.Using.resource(Files.newInputStream(path))(properties.load))
    sys.env.get(name).orElse(Option(properties.getProperty(name)))

  private def maxAbs(a: Array[Float],b: Array[Float]): Double =
    require(a.length==b.length)
    a.indices.iterator.map(i => math.abs(a(i).toDouble-b(i).toDouble)).foldLeft(0.0)(math.max)

  test("SD2 single-file loads as family sd2 and matches the diffusers 23-layer text encoder") {
    val single=setting("SCALATORCH_SD2_SINGLE_FILE")
    val directory=setting("SCALATORCH_SD2_DIFFUSERS_DIR")
    assume(single.nonEmpty && directory.nonEmpty,
      "set SCALATORCH_SD2_SINGLE_FILE and SCALATORCH_SD2_DIFFUSERS_DIR to the released SD2.1 artifacts")
    val components=SingleFileSd1Model.load(Path.of(single.get).toRealPath(),16777216L,()=>())
    assertEquals(components.family,"sd2")
    val reference=PretrainedClip.load(Path.of(directory.get).resolve("text_encoder"))
    val tokenizer=ClipFastTokenizer.loadDirectory(Path.of(directory.get).resolve("tokenizer"))
      .withNuiFallbackNormalization
    val text="a red barn beside a green field, clear morning light"
    val ids=tokenizer.encode(text)
    val tokens=Tensor.fromLongs(ids,Shape.from(Seq(1L,ids.length.toLong)))
    val a=components.clip.encode(tokens)
    val b=reference.encode(tokens)
    assertEquals(a.finalHidden.data.shape,b.finalHidden.data.shape)
    assert(maxAbs(a.finalHidden.data.toFloatArray,b.finalHidden.data.toFloatArray)<=1e-4,
      "single-file CLIP hidden states diverge from the diffusers text encoder")
    assertEquals(a.pooled.data.shape,b.pooled.data.shape)
    assert(maxAbs(a.pooled.data.toFloatArray,b.pooled.data.toFloatArray)<=1e-4,
      "single-file CLIP pooled state diverges from the diffusers text encoder")
  }

  test("SD2 single-file U-Net and VAE tensors match the diffusers directory") {
    val single=setting("SCALATORCH_SD2_SINGLE_FILE")
    val directory=setting("SCALATORCH_SD2_DIFFUSERS_DIR")
    assume(single.nonEmpty && directory.nonEmpty,
      "set SCALATORCH_SD2_SINGLE_FILE and SCALATORCH_SD2_DIFFUSERS_DIR to the released SD2.1 artifacts")
    def compare(label: String,view: SingleFileTensorView,referenceDirectory: Path): Unit =
      def referenceName(name: String): String =
        if label!="VAE" then name else name.replace(".to_q.",".query.").replace(".to_k.",".key.")
          .replace(".to_v.",".value.").replace(".to_out.0.",".proj_attn.")
      scala.util.Using.resource(scalatorch.io.SafeTensorSet.open(referenceDirectory,filename="diffusion_pytorch_model.safetensors")) { reference =>
        assertEquals(view.tensors.keySet.map(referenceName),reference.tensors.keySet,s"$label mapped tensor names differ")
        val mismatches=view.tensors.keysIterator.toVector.sorted.flatMap { name =>
          val a=view.readFloat(name).toFloatArray
          val b=reference.readFloat(referenceName(name)).toFloatArray
          require(a.length==b.length,s"$label element count differs for $name")
          val difference=a.indices.iterator.map(i => math.abs(a(i)-b(i))).foldLeft(0f)(math.max)
          Option.when(difference!=0f)(s"$name(max=$difference)")
        }
        assert(mismatches.isEmpty,s"$label tensors differ: ${mismatches.take(8).mkString(", ")}")
      }
    scala.util.Using.resource(scalatorch.io.SafeTensorSet.openFile(Path.of(single.get).toRealPath())) { archive =>
      compare("U-Net",SingleFileTensorView.unet(archive),Path.of(directory.get).resolve("unet"))
      compare("VAE",SingleFileTensorView.vae(archive),Path.of(directory.get).resolve("vae"))
    }
  }

  test("SD2 single-file U-Net and VAE forward outputs match the diffusers directory") {
    val single=setting("SCALATORCH_SD2_SINGLE_FILE")
    val directory=setting("SCALATORCH_SD2_DIFFUSERS_DIR")
    assume(single.nonEmpty && directory.nonEmpty,
      "set SCALATORCH_SD2_SINGLE_FILE and SCALATORCH_SD2_DIFFUSERS_DIR to the released SD2.1 artifacts")
    val components=SingleFileSd1Model.load(Path.of(single.get).toRealPath(),16777216L,()=>())
    val root=Path.of(directory.get).toRealPath()
    val referenceUnet=PretrainedUNet.load(root.resolve("unet"))
    val latent=Tensor.fromFloats(Array.tabulate(4*8*8)(i => math.sin(i*.17).toFloat),Shape.from(Seq(1,4,8,8)))
    val context=Tensor.fromFloats(Array.tabulate(77*1024)(i => math.cos(i*.013).toFloat),Shape.from(Seq(1,77,1024)))
    val timestep=Tensor.fromFloats(Array(7.5f),Shape.from(Seq(1)))
    val a=components.unet(scalatorch.autograd.Variable(latent,false),timestep,
      scalatorch.autograd.Variable(context,false)).data.toFloatArray
    val b=referenceUnet(scalatorch.autograd.Variable(latent,false),timestep,
      scalatorch.autograd.Variable(context,false)).data.toFloatArray
    assert(maxAbs(a,b)<=1e-4,s"single-file U-Net output diverges from the diffusers model: ${maxAbs(a,b)}")
    val referenceVae=PretrainedVae.load(root.resolve("vae"))
    val decodedA=components.vae.decode(scalatorch.autograd.Variable(latent,false)).data.toFloatArray
    val decodedB=referenceVae.decode(scalatorch.autograd.Variable(latent,false)).data.toFloatArray
    assert(maxAbs(decodedA,decodedB)<=1e-4,
      s"single-file VAE output diverges from the diffusers model: ${maxAbs(decodedA,decodedB)}")
  }

  test("SD2 single-file and diffusers pipelines produce the same seeded latent") {
    val single=setting("SCALATORCH_SD2_SINGLE_FILE")
    val directory=setting("SCALATORCH_SD2_DIFFUSERS_DIR")
    assume(single.nonEmpty && directory.nonEmpty,
      "set SCALATORCH_SD2_SINGLE_FILE and SCALATORCH_SD2_DIFFUSERS_DIR to the released SD2.1 artifacts")
    val components=SingleFileSd1Model.load(Path.of(single.get).toRealPath(),16777216L,()=>())
    val root=Path.of(directory.get).toRealPath()
    val tokenizer=ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")).withNuiFallbackNormalization
    val singlePipeline=new EulerImagePipeline(tokenizer,components.clip,components.unet,components.vae)
    val referencePipeline=new EulerImagePipeline(tokenizer,PretrainedClip.load(root.resolve("text_encoder")),
      PretrainedUNet.load(root.resolve("unet")),PretrainedVae.load(root.resolve("vae")))
    val config=EulerConfig.fromJson("""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val schedule=config.noise.nuiUniformEuler(1)
    val request=DiffusionImageRequest("a red barn beside a green field, clear morning light","",64,64,4321,guidanceScale=7)
    val a=singlePipeline.generateLatents(request,schedule).toFloatArray
    val b=referencePipeline.generateLatents(request,schedule).toFloatArray
    assert(maxAbs(a,b)<=1e-4,s"single-file seeded latent diverges from the diffusers pipeline: ${maxAbs(a,b)}")
  }
