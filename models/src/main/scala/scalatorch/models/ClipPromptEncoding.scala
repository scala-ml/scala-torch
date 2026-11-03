package scalatorch.models

import scalatorch.*

private[models] final case class EncodedPromptChunk(hidden: Tensor[Float],pooled: Option[Tensor[Float]])

/** Shared NUI CLIP tokenization, emphasis, encoding and sequence assembly. */
private[models] object ClipPromptEncoding:
  private val contextLength=77

  def chunks(tokenizer: ClipFastTokenizer,text: String): Vector[PromptChunk] =
    require(tokenizer.maxLength==contextLength,"NUI CLIP prompt conditioning requires 77-token contexts")
    val comma=tokenizer.encode(",",false)
    DiffusionPrompt.chunks(DiffusionPrompt.parse(text),s => tokenizer.encode(s,false),
      tokenizer.bosId,tokenizer.eosId,tokenizer.padId,
      if comma.length==1 then Some(comma.head) else None)

  private def emphasize(hidden: Tensor[Float],weights: Vector[Double]): Tensor[Float] =
    require(hidden.shape.rank==3 && hidden.shape.dims.head==1 && hidden.shape.dims(1)==weights.size,
      "prompt weights do not match CLIP hidden states")
    require(weights.forall(_.isFinite),"nonfinite prompt weight")
    val values=hidden.toFloatArray
    val width=hidden.shape.dims.last.toInt
    val before=values.iterator.map(_.toDouble).sum/values.length
    val beforeScale=values.iterator.map(v=>math.abs(v.toDouble)).sum/values.length
    var index=0
    while index<values.length do
      values(index)=(values(index)*weights(index/width)).toFloat
      index+=1
    val after=values.iterator.map(_.toDouble).sum/values.length
    val afterScale=values.iterator.map(v=>math.abs(v.toDouble)).sum/values.length
    val threshold=java.lang.Math.ulp(1.0f).toDouble*8.0
    if beforeScale>0.0 && afterScale>0.0 && math.abs(before)>threshold*beforeScale &&
        math.abs(after)>threshold*afterScale then
      val scale=before/after
      require(scale.isFinite,"prompt emphasis produced a nonfinite normalization")
      index=0
      while index<values.length do
        values(index)=(values(index)*scale).toFloat
        index+=1
    require(values.forall(_.isFinite),"prompt emphasis produced nonfinite CLIP features")
    Tensor.fromFloats(values,hidden.shape)

  def encodeChunk(tokenizer: ClipFastTokenizer,encoder: ClipTextModel,chunk: PromptChunk,
      skipLastLayers: Int,normalizeSelected: Boolean,cancelled: () => Boolean,
      legacyEosOverride: Option[Long]=None): EncodedPromptChunk =
    VaeExecution.checkCancelled(cancelled)
    require(chunk.tokens.size==contextLength && chunk.weights.size==contextLength,"invalid CLIP prompt chunk")
    val tokens=Tensor.fromLongs(chunk.tokens.toArray,Shape.from(Seq(1L,contextLength.toLong)))
    val output=encoder.encode(tokens,skipLastLayers,legacyEosOverride)
    VaeExecution.checkCancelled(cancelled)
    val selected=if skipLastLayers==0 then output.finalHidden
      else if normalizeSelected then encoder.finalNorm(output.hidden)
      else output.hidden
    EncodedPromptChunk(emphasize(selected.data,chunk.weights),output.projected.map(_.data))

  def concatenate(chunks: Seq[Tensor[Float]],maxElements: Long): Tensor[Float] =
    require(maxElements>=0 && chunks.nonEmpty,"invalid empty or bounded CLIP chunk sequence")
    val first=chunks.head
    require(first.shape.rank==3 && first.shape.dims.head==1 && first.shape.dims(1)==contextLength,
      "CLIP chunk must have shape (1,77,width)")
    val width=first.shape.dims(2)
    val sequence=chunks.size.toLong*contextLength
    val elements=sequence*width
    require(elements<=maxElements && elements<=Int.MaxValue,"CLIP conditioning exceeds allocation limit")
    val result=new Array[Float](elements.toInt)
    chunks.zipWithIndex.foreach { (chunk,index) =>
      require(chunk.shape==first.shape && chunk.dtype==first.dtype && chunk.device==first.device,
        "incompatible CLIP prompt chunks")
      val data=chunk.toFloatArray
      Array.copy(data,0,result,index*data.length,data.length)
    }
    Tensor.fromFloats(result,Shape.from(Seq(1L,sequence,width)))

  def padTo(input: Tensor[Float],targetChunks: Int,emptyChunk: Tensor[Float],maxElements: Long): Tensor[Float] =
    require(targetChunks>=0 && maxElements>=0,"invalid CLIP padding target or budget")
    require(input.shape.rank==3 && input.shape.dims.head==1 && input.shape.dims(1)%contextLength==0,
      "CLIP sequence must contain whole 77-token chunks")
    require(emptyChunk.shape==Shape.from(Seq(1L,contextLength.toLong,input.shape.dims(2))),
      "empty CLIP chunk shape mismatch")
    val current=(input.shape.dims(1)/contextLength).toInt
    require(targetChunks>=current,"cannot truncate CLIP sequence while padding")
    val sequence=targetChunks.toLong*contextLength
    val elements=sequence*input.shape.dims(2)
    require(elements<=maxElements && elements<=Int.MaxValue,"padded CLIP conditioning exceeds allocation limit")
    if current==targetChunks then input
    else
      val result=new Array[Float](elements.toInt)
      val data=input.toFloatArray
      val empty=emptyChunk.toFloatArray
      Array.copy(data,0,result,0,data.length)
      var offset=data.length
      while offset<result.length do
        Array.copy(empty,0,result,offset,empty.length)
        offset+=empty.length
      Tensor.fromFloats(result,Shape.from(Seq(1L,sequence,input.shape.dims(2))))
