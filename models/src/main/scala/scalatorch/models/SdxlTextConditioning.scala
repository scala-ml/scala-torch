package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Autograd

final case class SdxlPromptPair(positive: Tensor[Float],negative: Tensor[Float],
    pooled: Tensor[Float],negativePooled: Tensor[Float])

/** Base SDXL dual CLIP conditioning with NUI attention and chunk semantics. */
final class SdxlTextConditioning(val tokenizer: ClipFastTokenizer,val encoder: ClipTextModel,
    val tokenizer2: ClipFastTokenizer,val encoder2: ClipTextModel,val maxElements: Long=67108864L):
  require(maxElements>=0,"negative SDXL conditioning budget")
  require(encoder2.textProjection.nonEmpty,"SDXL encoder2 requires a text projection")
  private val width1=encoder.finalNorm.width
  private val width2=encoder2.finalNorm.width
  private val width=width1+width2
  val contextWidth: Int=width
  val pooledWidth: Int=encoder2.textProjection.get.weight.data.shape.dims.head.toInt
  for (tok,model) <- Seq(tokenizer->encoder,tokenizer2->encoder2) do
    require(tok.maxLength==77 && model.positionEmbedding.weight.data.shape.dims.head>=77,
      "SDXL conditioning requires 77-token CLIP contexts")
    require(tok.vocabSize==model.tokenEmbedding.weight.data.shape.dims.head,"SDXL tokenizer/encoder vocabulary mismatch")

  private def encodeText(text: String,skip: Int,cancelled: () => Boolean): (Tensor[Float],Tensor[Float]) =
    VaeExecution.checkCancelled(cancelled)
    val first=ClipPromptEncoding.chunks(tokenizer,text)
    val second=ClipPromptEncoding.chunks(tokenizer2,text)
    val count=math.max(first.size,second.size)
    val elements=count.toLong*77*width
    require(elements<=maxElements && elements<=Int.MaxValue,"SDXL conditioning budget exceeded")
    val output=new Array[Float](elements.toInt)
    var pooled: Option[Tensor[Float]]=None
    for index <- 0 until count do
      VaeExecution.checkCancelled(cancelled)
      val a=first(math.min(index,first.size-1));val b=second(math.min(index,second.size-1))
      val left=ClipPromptEncoding.encodeChunk(tokenizer,encoder,a,skip,normalizeSelected=false,cancelled=cancelled)
      val right=ClipPromptEncoding.encodeChunk(tokenizer2,encoder2,b,skip,normalizeSelected=false,cancelled=cancelled,
        legacyEosOverride=Some(tokenizer2.eosId))
      if pooled.isEmpty then pooled=right.pooled
      val x=left.hidden.toFloatArray;val y=right.hidden.toFloatArray
      for token <- 0 until 77 do
        val offset=(index*77+token)*width
        Array.copy(x,token*width1,output,offset,width1)
        Array.copy(y,token*width2,output,offset+width1,width2)
    (Tensor.fromFloats(output,Shape.from(Seq(1L,count.toLong*77,width.toLong))),pooled.get)

  def encode(prompt: String,negativePrompt: String,clipSkip: Int=1,cancelled: () => Boolean=() => false): SdxlPromptPair = Autograd.noGrad {
    VaeExecution.checkCancelled(cancelled)
    require(Seq(prompt,negativePrompt).forall(s => s.length<=1048576 && TokenizerJson.validUnicode(s)),"invalid or oversized SDXL prompt")
    require(clipSkip>=1,"SDXL clip skip must be positive")
    val skip=math.max(clipSkip,2)-1
    require(skip<=encoder.layers.size && skip<=encoder2.layers.size,"SDXL clip skip exceeds encoder depth")
    val (positive,pooled)=encodeText(prompt,skip,cancelled)
    // Python str.strip also considers Unicode space separators and U+001C..U+001F whitespace.
    val blank=negativePrompt.codePoints().allMatch(cp => Character.isWhitespace(cp) || Character.isSpaceChar(cp) || cp==0x85 || (cp>=28 && cp<=31))
    if blank then
      SdxlPromptPair(positive,Tensor.fromFloats(new Array[Float](positive.shape.numel.toInt),positive.shape),
        pooled,Tensor.fromFloats(new Array[Float](pooled.shape.numel.toInt),pooled.shape))
    else
      val (negative,negativePooled)=encodeText(negativePrompt,skip,cancelled)
      val length=math.max(positive.shape.dims(1),negative.shape.dims(1))
      if positive.shape==negative.shape then SdxlPromptPair(positive,negative,pooled,negativePooled)
      else
        val (empty,_)=encodeText("",skip,cancelled)
        val chunks=(length/77).toInt
        val emptyChunk=empty.slice(1,0,77)
        SdxlPromptPair(ClipPromptEncoding.padTo(positive,chunks,emptyChunk,maxElements),
          ClipPromptEncoding.padTo(negative,chunks,emptyChunk,maxElements),pooled,negativePooled)
  }
