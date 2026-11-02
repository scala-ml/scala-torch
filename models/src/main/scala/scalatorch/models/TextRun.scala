package scalatorch.models

import java.nio.file.Path

/** Offline CPU text generation from an existing model directory. */
object TextRun:
  def main(args: Array[String]): Unit =
    require(args.length >= 2 && args.length <= 4,
      "usage: TextRun MODEL_DIRECTORY PROMPT [MAX_NEW_TOKENS [--smollm-chat]]")
    require(args.length < 4 || args(3) == "--smollm-chat", "unsupported text generation option")
    val directory = Path.of(args(0))
    val count = if args.length >= 3 then args(2).toInt else 32
    require(count >= 0, "negative max new tokens")
    val prompt = if args.length == 4 then SmolChat.format(Vector(ChatMessage("user",args(1)))) else args(1)
    val tokenizer = ByteBpeTokenizer.read(directory.resolve("tokenizer.json"))
    val model = LlamaModel.load(directory)
    println(TextGeneration.generate(model,tokenizer,prompt,count).text)
