package scalatorch.models

final case class GeneratedText(ids: Vector[Long], text: String)

object TextGeneration:
  /** Greedy continuation; the caller supplies any desired chat template or special tokens. */
  def generate(model: LlamaModel, tokenizer: ByteBpeTokenizer, prompt: String, maxNewTokens: Int): GeneratedText =
    require(tokenizer.vocabSize == model.config.vocabSize, "tokenizer/model vocabulary size mismatch")
    val input = tokenizer.encode(prompt).toVector
    require(input.nonEmpty, "prompt must encode to at least one token")
    val continuation = model.generateIds(input, maxNewTokens).drop(input.size)
    GeneratedText(continuation, tokenizer.decode(continuation.toArray))

final case class ChatMessage(role: String, content: String)

/** Explicit formatter for the pinned SmolLM2 Instruct template, not a Jinja interpreter. */
object SmolChat:
  val defaultSystem: String = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
  def format(messages: Vector[ChatMessage], addGenerationPrompt: Boolean = true): String =
    require(messages.forall(m => Set("system", "user", "assistant")(m.role)), "unsupported chat role")
    val withSystem = if messages.nonEmpty && messages.head.role != "system" then
      ChatMessage("system", defaultSystem) +: messages else messages
    withSystem.map(m => s"<|im_start|>${m.role}\n${m.content}<|im_end|>\n").mkString +
      (if addGenerationPrompt then "<|im_start|>assistant\n" else "")
