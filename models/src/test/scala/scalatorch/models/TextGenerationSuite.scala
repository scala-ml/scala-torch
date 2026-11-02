package scalatorch.models

import munit.FunSuite
import java.nio.file.Path

class TextGenerationSuite extends FunSuite:
  test("Smol chat formatting matches default system and explicit history") {
    val default = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
    assertEquals(SmolChat.format(Vector(ChatMessage("user", "Hello!"))),
      default + "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")
    assertEquals(SmolChat.format(Vector(ChatMessage("system", "Be brief."), ChatMessage("user", "Hi"), ChatMessage("assistant", "Hello")), false),
      "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n")
    assertEquals(SmolChat.format(Vector.empty), "<|im_start|>assistant\n")
    intercept[IllegalArgumentException](SmolChat.format(Vector(ChatMessage("tool", "unsupported"))))
  }
  test("text generation returns only continuation and validates model vocabulary and prompt") {
    val source = scala.io.Source.fromResource("tokenizer/synthetic.json")
    val tokenizer = try ByteBpeTokenizer.fromJson(source.mkString) finally source.close()
    val config = LlamaConfig(tokenizer.vocabSize, 4, 4, 1, 1, 1, maxPositions=16,
      eosTokenIds=Vector.empty, initializerRange=0)
    val model = LlamaModel.init(config, 1, trainable=false)
    val result = TextGeneration.generate(model,tokenizer,"Hi",2)
    assertEquals(result.ids, Vector(0L,0L))
    val sampled = TextGeneration.generate(model,tokenizer,"Hi",2,SamplingConfig(temperature=1,topK=2,seed=42))
    assertEquals(sampled.ids,Vector(1L,1L))
    assertEquals(sampled.text,"\u0000\u0000")
    assertEquals(result.text, "") // ID0 is special and skipped by decode.
    assertEquals(TextGeneration.generate(model,tokenizer,"Hi",0), GeneratedText(Vector.empty,""))
    intercept[IllegalArgumentException](TextGeneration.generate(model,tokenizer,"",1))
    val other = model.copy(config=config.copy(vocabSize=config.vocabSize-1))
    intercept[IllegalArgumentException](TextGeneration.generate(other,tokenizer,"Hi",1))
  }
