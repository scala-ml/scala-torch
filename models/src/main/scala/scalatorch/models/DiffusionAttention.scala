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
    if context.isEmpty && !input.requiresGrad && Pag.isCurrentLayerPerturbed then
      val batch=input.data.shape.dims(0);val queries=input.data.shape.dims(1)
      val v=toV(input.reshape(Shape.from(Seq(batch*queries,queryWidth))))
      return toOut(v).reshape(Shape.from(Seq(batch,queries,queryWidth)))
    if context.isEmpty && !input.requiresGrad then
      HyperTile.current.flatMap(_.split(input.data.shape.dims(1).toInt)) match
        case Some((nh,nw,h,w)) =>
          val b=input.data.shape.dims(0).toInt
          val tiled=Variable(HyperTile.tileTokens(input.data,nh,nw,h,w),requiresGrad=false)
          val out=applyRaw(tiled,None)
          return Variable(HyperTile.untileTokens(out.data,b,nh,nw,h,w),requiresGrad=false)
        case None => ()
      val tdCtx=for
        td <- TokenDownsampling.current
        (sh,sw) <- TokenDownsampling.currentSpatialShape
        factor = td.factorForSection(Pag.currentSection)
        if factor > 1.0 && sh.toLong * sw == input.data.shape.dims(1)
      yield Variable(TokenDownsampling.resampleTokens(input.data,sh,sw,factor),requiresGrad=false)
      if tdCtx.isDefined then return applyRaw(input,tdCtx)
      val aaResult=for
        aa <- AgentAttention.current
        (sh,sw) <- TokenDownsampling.currentSpatialShape
        tokens = input.data.shape.dims(1).toInt
        if sh * sw == tokens && AgentAttention.downsampleOf(aa.latentH,aa.latentW,tokens) <= aa.settings.maxDownsample
        r = (tokens * aa.settings.ratio).toInt
        if r > 0 && sh >= aa.settings.sy && sw >= aa.settings.sx
      yield
        val b=input.data.shape.dims(0).toInt
        val seedOpt=if b%2==0 then Some(aa.seed) else None
        val plan=AgentAttention.planMerge(input.data,sh,sw,aa.settings.sx,aa.settings.sy,r,(tokens*aa.settings.agentRatio).toInt,seedOpt,aa.settings.useFp32)
        val (features,agentsOpt)=AgentAttention.merge(plan,input.data)
        val mergedOut=agentsOpt match
          case Some(ag) if ag.shape.dims(1)*2 < features.shape.dims(1) =>
            AgentAttention.applyAgentAttention(this,features,ag)
          case _ =>
            applyRaw(Variable(features,requiresGrad=false),None).data
        Variable(AgentAttention.unmerge(plan,mergedOut),requiresGrad=false)
      if aaResult.isDefined then return aaResult.get
      val refResult=ReferenceOnly.current.flatMap { refState =>
        val key=refState.nextLayerKey(Pag.currentSection)
        if refState.config.attentionAutoMachineWeight >= ReferenceOnly.autoMachineWeightForSection(Pag.currentSection) then
          refState.mode match
            case ReferenceOnly.Mode.Write =>
              val modIn=ControlLLLite.current.fold(input.data)(st => ControlLLLite.modulateAttentionInput(input.data,st))
              refState.bank.update(key,modIn)
              None
            case ReferenceOnly.Mode.ReadCond =>
              refState.bank.get(key).map { refTokens =>
                val modIn=ControlLLLite.current.fold(input)(st => Variable(ControlLLLite.modulateAttentionInput(input.data,st),requiresGrad=false))
                val catCtx=Variable(ReferenceOnly.concatTokens(modIn.data,refTokens),requiresGrad=false)
                applyRaw(modIn,Some(catCtx))
              }
            case ReferenceOnly.Mode.ReadUncond =>
              refState.bank.get(key).map { refTokens =>
                val modIn=ControlLLLite.current.fold(input)(st => Variable(ControlLLLite.modulateAttentionInput(input.data,st),requiresGrad=false))
                val catCtx=Variable(ReferenceOnly.concatTokens(modIn.data,refTokens),requiresGrad=false)
                val withRef=applyRaw(modIn,Some(catCtx))
                if refState.config.styleFidelity <= 0.0 then withRef
                else
                  val plain=applyRaw(modIn,None)
                  Variable(ReferenceOnly.blendUncond(withRef.data,plain.data,refState.config.styleFidelity),requiresGrad=false)
              }
        else None
      }
      if refResult.isDefined then return refResult.get
    val modInput=if !input.requiresGrad then ControlLLLite.current.fold(input)(st => Variable(ControlLLLite.modulateAttentionInput(input.data,st),requiresGrad=false)) else input
    val modContext=if context.isDefined && !input.requiresGrad then
      IpAdapter.current.fold(context)(ip => context.map(c => Variable(IpAdapter.augmentCrossAttentionContext(c.data,ip),requiresGrad=false)))
    else context
    applyRaw(modInput,modContext)

  private def applyRaw(input: Variable,context: Option[Variable]): Variable =
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
    val qRaw=project(input,toQ,queries);val k=project(source,toK,keys)
    val q=if context.isEmpty && !input.requiresGrad && Pag.currentSection=="mid" then
      Sag.recordFromQK(qRaw.data,k.data)
      Incantations.currentSeg.fold(qRaw) { case (sigma,ih,iw) => Variable(Incantations.segBlurQuery(qRaw.data,sigma,ih,iw),requiresGrad=false) }
    else qRaw
    val vRaw=project(source,toV,keys)
    val v=if context.isDefined && !input.requiresGrad then
      NegPip.currentSigns.fold(vRaw)(signs => Variable(NegPip.applySigns(vRaw.data,signs),requiresGrad=false))
    else vRaw
    val attended=q.scaledDotProductAttention(k,v,maxScoreElements).transpose(1,2)
      .reshape(Shape.from(Seq(batch*queries,inner)))
    toOut(attended).reshape(Shape.from(Seq(batch,queries,queryWidth)))
