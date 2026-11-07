package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Autograd

final case class SingleClipPromptPair(positive: Tensor[Float],negative: Option[Tensor[Float]])

/** Standard SD1.x single-CLIP prompt conditioning with NUI attention and chunk behavior. */
final class SingleClipTextConditioning(tokenizer: ClipFastTokenizer,encoder: ClipTextModel,
    maxElements: Long=67108864L):
  require(maxElements>=0,"negative single-CLIP conditioning budget")
  require(tokenizer.maxLength==77 && encoder.positionEmbedding.weight.data.shape.dims(0)>=77,
    "single-CLIP NUI conditioning requires a 77-token context")
  require(tokenizer.vocabSize==encoder.tokenEmbedding.weight.data.shape.dims(0),
    "single-CLIP tokenizer/encoder vocabulary mismatch")

  private def encodeText(text: String,clipSkip: Int,truncatePrompt: Boolean,
      cancelled: () => Boolean): Vector[Tensor[Float]] =
    val promptChunks=ClipPromptEncoding.chunks(tokenizer,text)
    val selected=if truncatePrompt then promptChunks.take(1) else promptChunks
    val elements=selected.size.toLong*77*encoder.finalNorm.width
    require(elements<=maxElements && elements<=Int.MaxValue,"single-CLIP conditioning exceeds allocation limit")
    val skip=clipSkip-1
    selected.map { chunk =>
      ClipPromptEncoding.encodeChunk(tokenizer,encoder,chunk,skip,normalizeSelected=true,cancelled).hidden
    }

  def encode(prompt: String,negativePrompt: String,clipSkip: Int=1,
      truncatePrompt: Boolean=false,encodeNegative: Boolean=true,
      cancelled: () => Boolean=() => false): SingleClipPromptPair = Autograd.noGrad {
    VaeExecution.checkCancelled(cancelled)
    require(clipSkip>=1 && clipSkip<=encoder.layers.size+1,"CLIP skip exceeds text encoder layer count")
    val positiveChunks=encodeText(prompt,clipSkip,truncatePrompt,cancelled)
    val positive=ClipPromptEncoding.concatenate(positiveChunks,maxElements)
    if !encodeNegative then SingleClipPromptPair(positive,None)
    else
      val negativeChunks=encodeText(negativePrompt,clipSkip,truncatePrompt,cancelled)
      val target=math.max(positiveChunks.size,negativeChunks.size)
      val empty=encodeText("",clipSkip,truncatePrompt,cancelled).head
      val negative=ClipPromptEncoding.concatenate(negativeChunks,maxElements)
      SingleClipPromptPair(
        ClipPromptEncoding.padTo(positive,target,empty,maxElements),
        Some(ClipPromptEncoding.padTo(negative,target,empty,maxElements)))
  }
