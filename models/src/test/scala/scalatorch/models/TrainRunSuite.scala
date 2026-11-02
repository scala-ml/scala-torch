package scalatorch.models

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

class TrainRunSuite extends munit.FunSuite:
  private def fixture[A](body: Path => A): A =
    val dir=Files.createTempDirectory("scala-training-cli-")
    try
      val input=getClass.getResourceAsStream("/tokenizer/synthetic.json")
      val tokenizer=try new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8) finally input.close()
      Files.writeString(dir.resolve("tokenizer.json"),tokenizer)
      val vocab=ByteBpeTokenizer.fromJson(tokenizer).vocabSize
      val config=s"""{"model_type":"llama","vocab_size":$vocab,"hidden_size":8,"intermediate_size":12,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16,"bos_token_id":1,"eos_token_id":${vocab-1},"tie_word_embeddings":true,"initializer_range":0.1}"""
      Files.writeString(dir.resolve("config.json"),config)
      val model=LlamaModel.init(LlamaConfig.fromJson(config),17)
      val values=model.namedWeights.filterNot(_._1=="lm_head.weight")
      val bytes=ByteBuffer.allocate(values.map(_._2.data.shape.numel.toInt*4).sum).order(ByteOrder.LITTLE_ENDIAN)
      var offset=0L
      val fields=values.map { (name,p) =>
        val start=offset;offset+=p.data.shape.numel*4
        p.data.toFloatArray.foreach(bytes.putFloat)
        s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${p.data.shape.dims.mkString(",")}],\"data_offsets\":[$start,$offset]}"
      }
      val header=("{"+fields.mkString(",")+"}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
      Files.write(dir.resolve("model.safetensors"),ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).array()++header++bytes.array())
      Files.writeString(dir.resolve("data.txt"),"ababa ababa\n日本語 ababa\nababa ababa\n")
      body(dir)
    finally Using.resource(Files.walk(dir)) { paths =>
      paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(path => Files.delete(path))
    }
  private def args(dir: Path,output: String,steps: Int,backend: String = "cpu",extra: Seq[String] = Seq.empty): Array[String] =
    (Seq(backend,dir.toString,dir.resolve("data.txt").toString,dir.resolve(output).toString,steps.toString,
      "--sequence-length","4","--batch-size","2","--learning-rate","0.02")++extra).toArray
  private def continuation(backend: String,full: Boolean): Unit = fixture { dir =>
    val mode=if full then Seq("--full") else Seq.empty
    val logs=ArrayBuffer.empty[String]
    TrainRun.run(TrainRun.parse(args(dir,"whole.stb",4,backend,mode)),line => { logs+=line;() })
    TrainRun.run(TrainRun.parse(args(dir,"resume.stb",2,backend,mode)),_ => ())
    TrainRun.run(TrainRun.parse(args(dir,"resume.stb",4,backend,mode :+ "--resume")),_ => ())
    assertEquals(Files.readAllBytes(dir.resolve("whole.stb")).toVector,Files.readAllBytes(dir.resolve("resume.stb")).toVector)
    assertEquals(logs.count(_.startsWith("step=")),4)
    assert(logs.exists(_.startsWith("saved_step=4")))
    val saved=Files.readAllBytes(dir.resolve("resume.stb")).toVector
    intercept[IllegalArgumentException](TrainRun.run(TrainRun.parse(args(dir,"resume.stb",5,backend,mode)),_ => ()))
    Files.writeString(dir.resolve("data.txt"),"different training data\n")
    intercept[IllegalArgumentException](TrainRun.run(TrainRun.parse(args(dir,"resume.stb",5,backend,mode :+ "--resume")),_ => ()))
    assertEquals(Files.readAllBytes(dir.resolve("resume.stb")).toVector,saved)
  }
  test("CPU CLI trains LoRA and full models with exact data-bound resume") {
    continuation("cpu",false);continuation("cpu",true)
  }
  test("CUDA CLI trains LoRA with exact data-bound resume") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
    continuation("cuda",false)
  }
  test("CLI rejects malformed, duplicate and conflicting options") {
    val base=Array("cpu","model","data","output","2")
    for suffix <- Seq(Seq("--unknown"),Seq("--sequence-length","0"),Seq("--batch-size","-1"),
        Seq("--learning-rate","NaN"),Seq("--resume","--resume"),Seq("--full","--lora-rank","4"),
        Seq("--lora-layers","1,1"),Seq("--save-every","0"),Seq("--device-bytes","0"),Seq("--seed")) do
      intercept[IllegalArgumentException](TrainRun.parse(base++suffix))
    intercept[IllegalArgumentException](TrainRun.parse(base.updated(0,"other")))
    intercept[IllegalArgumentException](TrainRun.parse(base.updated(4,"-1")))
    intercept[IllegalArgumentException](TrainRun.parse(Array.empty[String]))
  }
