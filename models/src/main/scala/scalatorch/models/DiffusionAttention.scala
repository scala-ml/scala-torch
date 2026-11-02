package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

/** CPU projected self/cross attention for (batch,tokens,channels), without masks or dropout. */
final case class DiffusionAttention(toQ: Linear,toK: Linear,toV: Linear,toOut: Linear,heads: Int,
    maxScoreElements: Long = 16777216L) derives Module:
  private val inner=toQ.weight.data.shape.dims.head
  private val queryWidth=toQ.weight.data.shape.dims(1)
  private val contextWidth=toK.weight.data.shape.dims(1)
  require(heads>0 && inner>0 && inner%heads==0 && queryWidth>0 && contextWidth>0,"invalid attention head/feature widths")
  require(toK.weight.data.shape.dims.head==inner && toV.weight.data.shape==toK.weight.data.shape &&
    toOut.weight.data.shape.dims(1)==inner && toOut.weight.data.shape.dims.head==queryWidth,"incompatible attention projections")
  require(maxScoreElements>=0,"invalid attention score budget")

  def apply(input: Variable,context: Option[Variable] = None): Variable =
    val source=context.getOrElse(input)
    require(input.data.shape.rank==3 && input.data.shape.dims(2)==queryWidth,"query must have matching (batch,tokens,channels)")
    require(source.data.shape.rank==3 && source.data.shape.dims(0)==input.data.shape.dims(0) &&
      source.data.shape.dims(2)==contextWidth,"context batch/channel mismatch")
    val batch=input.data.shape.dims(0);val queries=input.data.shape.dims(1);val keys=source.data.shape.dims(1)
    // Validate workspace before materializing projections.
    val scores=Shape.from(Seq(batch,heads.toLong,queries,keys)).numel
    require(scores<=maxScoreElements,"attention score workspace limit exceeded")
    def project(x: Variable,linear: Linear,tokens: Long): Variable =
      linear(x.reshape(Shape.from(Seq(batch*tokens,x.data.shape.dims(2)))))
        .reshape(Shape.from(Seq(batch,tokens,heads.toLong,inner/heads))).transpose(1,2)
    val q=project(input,toQ,queries);val k=project(source,toK,keys);val v=project(source,toV,keys)
    val attended=q.scaledDotProductAttention(k,v,maxScoreElements).transpose(1,2)
      .reshape(Shape.from(Seq(batch*queries,inner)))
    toOut(attended).reshape(Shape.from(Seq(batch,queries,queryWidth)))
