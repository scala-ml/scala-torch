package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{GeluApproximation,Variable}
import scalatorch.nn.*

enum ClipActivation:
  case QuickGelu, Gelu, GeluTanh
  def apply(input: Variable): Variable = this match
    case QuickGelu => input.quickGelu()
    case Gelu => input.gelu()
    case GeluTanh => input.gelu(GeluApproximation.Tanh)
object ClipActivation:
  given Module[ClipActivation] with
    def namedParameters(value: ClipActivation) = Vector.empty

/** Pre-norm CLIP text layer, with causal attention and no dropout or padding mask. */
final case class ClipTextLayer(norm1: LayerNorm,qProj: Linear,kProj: Linear,vProj: Linear,outProj: Linear,
    norm2: LayerNorm,fc1: Linear,fc2: Linear,heads: Int,activation: ClipActivation) derives Module:
  val width: Int=norm1.width
  require(heads>0 && width%heads==0 && norm2.width==width,"invalid CLIP layer norm/head dimensions")
  require(Vector(qProj,kProj,vProj,outProj).forall(_.weight.data.shape==Shape.from(Seq(width.toLong,width.toLong))),
    "CLIP attention projection shape mismatch")
  require(fc1.weight.data.shape.dims(1)==width && fc1.weight.data.shape.dims.head>0 &&
    fc2.weight.data.shape==Shape.from(Seq(width.toLong,fc1.weight.data.shape.dims.head)),"CLIP MLP shape mismatch")
  def apply(input: Variable,maxScoreElements: Long=16777216L): Variable =
    require(input.data.shape.rank==3 && input.data.shape.dims(2)==width,"CLIP layer expects matching token features")
    val b=input.data.shape.dims(0);val t=input.data.shape.dims(1)
    require(maxScoreElements>=0 && Shape.from(Seq(b,heads.toLong,t,t)).numel<=maxScoreElements,"CLIP attention budget exceeded")
    val normalized=norm1(input).reshape(Shape.from(Seq(b*t,width.toLong)))
    def project(linear: Linear): Variable=linear(normalized).reshape(Shape.from(Seq(b,t,heads.toLong,(width/heads).toLong))).transpose(1,2)
    val attended=project(qProj).causalAttention(project(kProj),project(vProj),maxScoreElements=maxScoreElements)
      .transpose(1,2).reshape(Shape.from(Seq(b*t,width.toLong)))
    val hidden=input+outProj(attended).reshape(input.data.shape)
    val flat=norm2(hidden).reshape(Shape.from(Seq(b*t,width.toLong)))
    hidden+fc2(activation(fc1(flat))).reshape(input.data.shape)

final case class ClipTextOutput(hidden: Variable,finalHidden: Variable,pooled: Variable,projected: Option[Variable])

/** apply preserves normalized SD1.x output; encode exposes raw selected states and pooled projection. */
final case class ClipTextModel(tokenEmbedding: Embedding,positionEmbedding: Embedding,layers: Vector[ClipTextLayer],
    finalNorm: LayerNorm,maxScoreElements: Long=16777216L,maxHiddenElements: Long=16777216L,
    textProjection: Option[Linear]=None,eosTokenId: Long=2L) derives Module:
  private val width=finalNorm.width
  require(layers.nonEmpty && layers.size<=512 && layers.forall(_.width==width),"invalid CLIP layer stack")
  require(tokenEmbedding.weight.data.shape.dims(1)==width && positionEmbedding.weight.data.shape.dims(1)==width &&
    tokenEmbedding.weight.data.shape.dims.head>0 && positionEmbedding.weight.data.shape.dims.head>0,"CLIP embedding shape mismatch")
  require(maxScoreElements>=0 && maxHiddenElements>=0,"negative CLIP workspace limit")
  require(eosTokenId>=0,"negative CLIP EOS ID")
  textProjection.foreach(p => require(p.weight.data.shape.dims(1)==width && p.bias.isEmpty,"invalid CLIP text projection"))
  /** Reorder vocabulary rows without mutating the original model. */
  def withSelectedTokenEmbeddings(ids: Vector[Long]): ClipTextModel =
    val current=tokenEmbedding.weight.data
    val count=current.shape.dims.head
    require(ids.nonEmpty && ids.size<=count && ids.distinct.size==ids.size && ids.forall(id=>id>=0 && id<count),
      "invalid CLIP vocabulary selection")
    require(ids.size.toLong*width<=Int.MaxValue,"CLIP vocabulary selection exceeds allocation limit")
    val original=current.toFloatArray
    val selected=new Array[Float](ids.size*width)
    ids.zipWithIndex.foreach((id,index)=>Array.copy(original,id.toInt*width,selected,index*width,width))
    copy(tokenEmbedding=Embedding(Parameter(Tensor.fromFloats(selected,Shape.from(Seq(ids.size.toLong,width.toLong))),
      tokenEmbedding.weight.trainable)))
  /** Replace selected rows in a separate model for a scoped preview. */
  def withReplacedTokenEmbeddings(ids: Vector[Long],rows: Vector[Array[Float]]): ClipTextModel =
    val tensor=tokenEmbedding.weight.data
    require(ids.nonEmpty && ids.size==rows.size && ids.distinct.size==ids.size &&
      ids.forall(id=>id>=0 && id<tensor.shape.dims.head),"invalid CLIP embedding row selection")
    require(rows.forall(row=>row!=null && row.length==width && row.forall(_.isFinite)),"invalid CLIP embedding rows")
    val values=tensor.toFloatArray
    ids.zip(rows).foreach((id,row)=>Array.copy(row,0,values,id.toInt*width,width))
    copy(tokenEmbedding=Embedding(Parameter(Tensor.fromFloats(values,tensor.shape),tokenEmbedding.weight.trainable)))
  def withAddedTokenEmbeddings(rows: Vector[Array[Float]]): ClipTextModel =
    require(rows.nonEmpty && rows.size<=256,"CLIP token extension must contain 1..256 rows")
    val current=tokenEmbedding.weight.data
    val vocabulary=current.shape.dims(0).toInt
    val rowWidth=current.shape.dims(1).toInt
    require(rows.forall(row => row.length==rowWidth && row.forall(_.isFinite)),
      "CLIP textual-inversion rows have invalid width or nonfinite values")
    require(vocabulary.toLong+rows.size<=Int.MaxValue && (vocabulary.toLong+rows.size)*rowWidth<=Int.MaxValue,
      "CLIP textual-inversion extension exceeds allocation limit")
    val original=current.toFloatArray
    val extended=java.util.Arrays.copyOf(original,original.length+rows.size*rowWidth)
    rows.zipWithIndex.foreach { (row,index) => Array.copy(row,0,extended,original.length+index*rowWidth,rowWidth) }
    copy(tokenEmbedding=Embedding(Parameter(Tensor.fromFloats(extended,
      Shape.from(Seq(vocabulary.toLong+rows.size,rowWidth.toLong))),tokenEmbedding.weight.trainable)))
  private def input(tokens: Tensor[Long],skipLastLayers: Int): Variable =
    require(tokens.shape.rank==2,"CLIP token IDs must have shape (batch,tokens)")
    val b=tokens.shape.dims(0);val t=tokens.shape.dims(1)
    require(t>0 && t<=positionEmbedding.weight.data.shape.dims.head && t<=Int.MaxValue,"CLIP context length exceeded or empty")
    require(skipLastLayers>=0 && skipLastLayers<=layers.size,"invalid CLIP skipped layer count")
    // Positional embeddings are materialized for one row, even when the input batch is empty.
    val positional=Shape.from(Seq(1L,t,width.toLong)).numel
    require(positional<=maxHiddenElements && positional<=Int.MaxValue,"CLIP position workspace limit exceeded")
    val hidden=Shape.from(Seq(b,t,width.toLong)).numel
    require(hidden<=maxHiddenElements && hidden<=Int.MaxValue,"CLIP hidden workspace limit exceeded")
    for layer <- layers.take(layers.size-skipLastLayers) do
      val scores=Shape.from(Seq(b,layer.heads.toLong,t,t)).numel
      require(scores<=maxScoreElements && scores<=Int.MaxValue,"CLIP attention budget exceeded")
      val mlp=Shape.from(Seq(b,t,layer.fc1.weight.data.shape.dims.head)).numel
      require(mlp<=maxHiddenElements && mlp<=Int.MaxValue,"CLIP MLP workspace limit exceeded")
    val positions=Tensor.fromLongs(Array.tabulate(t.toInt)(_.toLong),Shape.from(Seq(1L,t)))
    tokenEmbedding(tokens)+positionEmbedding(positions)

  def apply(tokens: Tensor[Long],skipLastLayers: Int=0): Variable =
    val embedded=input(tokens,skipLastLayers)
    val encoded=layers.take(layers.size-skipLastLayers).foldLeft(embedded)((x,layer) => layer(x,maxScoreElements))
    finalNorm(encoded)

  /** Raw state after N-skip layers, plus fully executed normalized/pooled output.
    * legacyEosOverride follows NUI's added-token fix only for legacy eosTokenId=2.
    */
  def encode(tokens: Tensor[Long],skipLastLayers: Int=1,legacyEosOverride: Option[Long]=None): ClipTextOutput =
    require(skipLastLayers>=0 && skipLastLayers<=layers.size,"invalid CLIP skipped layer count")
    require(legacyEosOverride.forall(_>=0),"negative legacy EOS override")
    require(tokens.shape.rank==2,"CLIP token IDs must have shape (batch,tokens)")
    textProjection.foreach { p =>
      val projected=Shape.from(Seq(tokens.shape.dims.head,p.weight.data.shape.dims.head)).numel
      require(projected<=maxHiddenElements && projected<=Int.MaxValue,"CLIP projected workspace limit exceeded")
    }
    var hidden=input(tokens,0)
    var selected=hidden
    val selectedDepth=layers.size-skipLastLayers
    layers.zipWithIndex.foreach { (layer,index) =>
      hidden=layer(hidden,maxScoreElements)
      if index+1==selectedDepth then selected=hidden
    }
    val normalized=finalNorm(hidden)
    val ids=tokens.toLongArray
    val batch=tokens.shape.dims.head.toInt;val count=tokens.shape.dims(1).toInt
    val rows=Vector.tabulate(batch) { row =>
      var maximum=0
      var firstEos= -1
      var overrideEos= -1
      for col <- 0 until count do
        val id=ids(row*count+col)
        if id>ids(row*count+maximum) then maximum=col
        if id==eosTokenId && firstEos<0 then firstEos=col
        if legacyEosOverride.contains(id) && overrideEos<0 then overrideEos=col
      val position=if eosTokenId!=2 then math.max(0,firstEos)
        else if overrideEos>=0 && ids(row*count+maximum)>legacyEosOverride.get then overrideEos
        else maximum
      normalized.narrow(0,row,1).narrow(1,position,1).reshape(Shape.from(Seq(1L,width.toLong)))
    }
    val pooled=if rows.isEmpty then normalized.narrow(1,0,1).reshape(Shape.from(Seq(0L,width.toLong))) else Variable.cat(rows,0)
    ClipTextOutput(selected,normalized,pooled,textProjection.map(_(pooled)))
