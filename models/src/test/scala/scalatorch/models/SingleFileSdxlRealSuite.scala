package scalatorch.models

import java.nio.file.{Files,Path}
import scalatorch.Shape

/** Opt-in real-checkpoint verification for SDXL 1.0 single-file loading against the released diffusers directory. */
class SingleFileSdxlRealSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(15,"minutes")

  private def setting(name: String): Option[String] =
    val properties=new java.util.Properties()
    val config=Vector(Path.of(".superpowers/sdxl-single-file-test.properties"),
      Path.of("../.superpowers/sdxl-single-file-test.properties")).find(Files.isRegularFile(_))
    config.foreach(path => scala.util.Using.resource(Files.newInputStream(path))(properties.load))
    sys.env.get(name).orElse(Option(properties.getProperty(name)))

  private def maxAbs(a: Array[Float],b: Array[Float]): Double =
    require(a.length==b.length)
    a.indices.iterator.map(i => math.abs(a(i).toDouble-b(i).toDouble)).foldLeft(0.0)(math.max)

  test("SDXL single-file CLIP-L, CLIP-G, U-Net and VAE views match the diffusers directory") {
    val single=setting("SCALATORCH_SDXL_SINGLE_FILE")
    val directory=setting("SCALATORCH_SDXL_DIFFUSERS_DIR")
    assume(single.nonEmpty && directory.nonEmpty,
      "set SCALATORCH_SDXL_SINGLE_FILE and SCALATORCH_SDXL_DIFFUSERS_DIR to the SDXL 1.0 artifacts")
    val root=Path.of(directory.get).toRealPath()
    val singlePath=SdxlSingleFileConverter.ensureSingleFile(root,Path.of(single.get))
    def compare(label: String,view: SingleFileTensorView,referenceDirectory: Path,filename: String): Unit =
      def referenceName(name: String): String =
        if label!="VAE" then name else name.replace(".to_q.",".query.").replace(".to_k.",".key.")
          .replace(".to_v.",".value.").replace(".to_out.0.",".proj_attn.")
      scala.util.Using.resource(scalatorch.io.SafeTensorSet.open(referenceDirectory,filename=filename)) { reference =>
        val mappedKeys=view.tensors.keySet.map { n =>
          if reference.tensors.contains(n) then n else referenceName(n)
        }
        assertEquals(mappedKeys,reference.tensors.keySet,s"$label mapped tensor names differ")
        val mismatches=view.tensors.keysIterator.toVector.sorted.flatMap { name =>
          val refName=if reference.tensors.contains(name) then name else referenceName(name)
          val a=view.readFloat(name).toFloatArray
          val b=reference.readFloat(refName).toFloatArray
          require(a.length==b.length,s"$label element count differs for $name")
          val difference=a.indices.iterator.map(i => math.abs(a(i)-b(i))).foldLeft(0f)(math.max)
          Option.when(difference!=0f)(s"$name(max=$difference)")
        }
        assert(mismatches.isEmpty,s"$label tensors differ: ${mismatches.take(8).mkString(", ")}")
      }
    scala.util.Using.resource(scalatorch.io.SafeTensorSet.openFile(singlePath)) { archive =>
      SingleFileSdxlConfig.inspect(archive)
      compare("CLIP-L",SingleFileTensorView.sdxlClip(archive),root.resolve("text_encoder"),"model.safetensors")
      compare("CLIP-G",SingleFileTensorView.sdxlClip2(archive),root.resolve("text_encoder_2"),"model.safetensors")
      compare("U-Net",SingleFileTensorView.unet(archive),root.resolve("unet"),"diffusion_pytorch_model.safetensors")
      compare("VAE",SingleFileTensorView.vae(archive),root.resolve("vae"),"diffusion_pytorch_model.safetensors")
    }
  }

  test("SDXL single-file loads as family sdxl and matches dual-CLIP conditioning and VAE decode") {
    val single=setting("SCALATORCH_SDXL_SINGLE_FILE")
    val directory=setting("SCALATORCH_SDXL_DIFFUSERS_DIR")
    assume(single.nonEmpty && directory.nonEmpty,
      "set SCALATORCH_SDXL_SINGLE_FILE and SCALATORCH_SDXL_DIFFUSERS_DIR to the SDXL 1.0 artifacts")
    val root=Path.of(directory.get).toRealPath()
    val singlePath=SdxlSingleFileConverter.ensureSingleFile(root,Path.of(single.get))
    val latent=scalatorch.Tensor.fromFloats(Array.tabulate(4*8*8)(i => math.sin(i*.17).toFloat),Shape.from(Seq(1,4,8,8)))
    val (a,decodedA) = locally {
      val components=SingleFileSd1Model.load(singlePath,16777216L,()=>())
      assertEquals(components.family,"sdxl")
      assert(components.clip2.isDefined,"SDXL single-file must provide clip2")
      val tokenizer=ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")).withNuiFallbackNormalization
      val tokenizer2=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),
        root.resolve("tokenizer/merges.txt"),padToken="!").withNuiFallbackNormalization
      val singleCond=new SdxlTextConditioning(tokenizer,components.clip,tokenizer2,components.clip2.get)
      val cond=singleCond.encode("a photograph of a red apple on a wooden table","blurry")
      val vaeOut=components.vae.decodeDiffusionLatents(scalatorch.autograd.Variable(latent,false)).data.toFloatArray
      (cond,vaeOut)
    }

    val refCond=new SdxlTextConditioning(
      ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")).withNuiFallbackNormalization,
      PretrainedClip.load(root.resolve("text_encoder")),
      ClipFastTokenizer.loadDirectory(root.resolve("tokenizer_2")).withNuiFallbackNormalization,
      PretrainedClip.load(root.resolve("text_encoder_2")))
    val b=refCond.encode("a photograph of a red apple on a wooden table","blurry")
    assertEquals(a.positive.shape,Shape.from(Seq(1,77,2048)))
    assertEquals(a.pooled.shape,Shape.from(Seq(1,1280)))
    assert(maxAbs(a.positive.toFloatArray,b.positive.toFloatArray)<=1e-5,
      "SDXL single-file positive conditioning diverges from diffusers directory")
    assert(maxAbs(a.negative.toFloatArray,b.negative.toFloatArray)<=1e-5,
      "SDXL single-file negative conditioning diverges from diffusers directory")
    assert(maxAbs(a.pooled.toFloatArray,b.pooled.toFloatArray)<=1e-5,
      "SDXL single-file pooled conditioning diverges from diffusers directory")
    assert(maxAbs(a.negativePooled.toFloatArray,b.negativePooled.toFloatArray)<=1e-5,
      "SDXL single-file negativePooled conditioning diverges from diffusers directory")

    val refVae=PretrainedVae.load(root.resolve("vae"))
    val decodedB=refVae.decodeDiffusionLatents(scalatorch.autograd.Variable(latent,false)).data.toFloatArray
    assert(maxAbs(decodedA,decodedB)<=1e-4,
      s"SDXL single-file VAE output diverges from diffusers directory: ${maxAbs(decodedA,decodedB)}")
  }
