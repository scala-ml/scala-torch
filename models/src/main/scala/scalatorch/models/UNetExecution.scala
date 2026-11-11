package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.ffm.CudaSession
import scala.util.Using

private[models] trait UNetPredictor:
  def apply(sample: Tensor[Float],timesteps: Tensor[Float],guidance: Float,controlProgress: Double=0.0,rescale: Float=0f,scheduled: Option[Float]=None,
      detail: Option[DetailDaemon.Adjustment]=None,controlProgressEnd: Double = -1.0,
      dynamicThresholding: Option[DynamicThresholdingConfig]=None,
      prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
      cads: Option[(CadsState,Double)]=None,
      guidanceHook: Option[GuidanceHookState]=None,
      pag: Option[PagConfig]=None,
      conditionalOverride: Option[Tensor[Float]]=None,
      positiveAddedOverride: Option[DiffusionAddedConditioning]=None,
      unconditionalOverride: Option[Tensor[Float]]=None,
      negativeAddedOverride: Option[DiffusionAddedConditioning]=None,
      positiveSigns: Option[Array[Float]]=None,
      negativeSigns: Option[Array[Float]]=None,
      kohyaHrFix: Option[KohyaHrFixConfig]=None,
      tokenDownsampling: Option[(TokenDownsamplingConfig,Boolean,Int,Int)]=None,
      multiDiffusion: Option[MultiDiffusionConfig]=None,
      segUncond: Option[(Double,Int,Int)]=None,
      chgState: Option[(CharacteristicGuidance.State,Int)]=None,
      neutralPlan: Option[NeutralPrompt.StepPlan]=None,
      sagStep: Option[(SagConfig,Double)]=None,
      refStep: Option[(ReferenceOnly.BankState,Tensor[Float])]=None): Tensor[Float]

/** A sample batch of several images met a step that only the single-image path implements; the caller samples image by image instead. */
final class BatchedSamplingUnsupported(message: String) extends IllegalStateException(message)

/** Placement of U-Net prediction. CUDA weights/contexts live for one denoising loop;
  * CPU schedulers exchange only latent inputs and guided predictions each step.
  */
sealed trait UNetExecution:
  def offloadWeights: Boolean=false
  def device: String
  private[models] def withPredictor[A](model: ConditionalUNet2d,conditional: Tensor[Float],
      unconditional: Option[Tensor[Float]],cancelled: () => Boolean,
      positiveAdded: Option[DiffusionAddedConditioning]=None,negativeAdded: Option[DiffusionAddedConditioning]=None,
      controlNets: Vector[DiffusionControlNetRequest]=Vector.empty)(body: UNetPredictor => A): A

object UNetExecution:
  private[models] def controlInput(sample: Tensor[Float]): Tensor[Float] =
    require(sample.shape.rank==4 && (sample.shape.dims(1)==4 || sample.shape.dims(1)==9),
      "ControlNet requires a four-channel latent or nine-channel inpaint sample")
    if sample.shape.dims(1)==4 then sample else sample.slice(1,0,4).contiguous

  private def guided(negative: Variable,positive: Variable,base: Float,rescale: Float,scheduled: Option[Float]): Tensor[Float] =
    val adjusted=scheduled.filter(_!=base).fold(positive)(target=>negative+(positive-negative)*(target/base))
    val result=ClassifierFreeGuidance.combine(negative,adjusted,base)
    if rescale!=0 then GuidanceRescale(result.data,adjusted.data,rescale) else result.data
  case object Cpu extends UNetExecution:
    val device="cpu"
    private[models] def withPredictor[A](model: ConditionalUNet2d,conditional: Tensor[Float],
        unconditional: Option[Tensor[Float]],cancelled: () => Boolean,
        positiveAdded: Option[DiffusionAddedConditioning]=None,negativeAdded: Option[DiffusionAddedConditioning]=None,
        controlNets: Vector[DiffusionControlNetRequest]=Vector.empty)(body: UNetPredictor => A): A =
      VaeExecution.checkCancelled(cancelled)
      require(unconditional.isDefined || negativeAdded.isEmpty,"negative addition requires negative context")
      body(new UNetPredictor:
        def apply(sample: Tensor[Float],timesteps: Tensor[Float],guidance: Float,
            controlProgress: Double,rescale: Float,scheduled: Option[Float],detail: Option[DetailDaemon.Adjustment],controlProgressEnd: Double,
            dynamicThresholding: Option[DynamicThresholdingConfig],prediction: DiffusionPrediction,cads: Option[(CadsState,Double)],
            guidanceHook: Option[GuidanceHookState],pag: Option[PagConfig],
            conditionalOverride: Option[Tensor[Float]],positiveAddedOverride: Option[DiffusionAddedConditioning],
            unconditionalOverride: Option[Tensor[Float]],negativeAddedOverride: Option[DiffusionAddedConditioning],
            positiveSigns: Option[Array[Float]],negativeSigns: Option[Array[Float]],
            kohyaHrFix: Option[KohyaHrFixConfig],tokenDownsampling: Option[(TokenDownsamplingConfig,Boolean,Int,Int)],
            multiDiffusion: Option[MultiDiffusionConfig],segUncond: Option[(Double,Int,Int)],
            chgState: Option[(CharacteristicGuidance.State,Int)],
            neutralPlan: Option[NeutralPrompt.StepPlan],
            sagStep: Option[(SagConfig,Double)],
            refStep: Option[(ReferenceOnly.BankState,Tensor[Float])]): Tensor[Float] = Autograd.noGrad {
          KohyaHrFix.withStep(kohyaHrFix,timesteps.toFloatArray(0)) {
          val tdStep=tokenDownsampling.getOrElse((TokenDownsamplingConfig(),false,0,1))
          TokenDownsampling.withStep(tokenDownsampling.map(_._1),tdStep._2,tdStep._3,tdStep._4,model.downBlocks.size) {
          VaeExecution.checkCancelled(cancelled)
          require(guidance.isFinite && guidance>=0,"invalid guidance scale")
          val active=controlNets.flatMap(request=>request.scales(controlProgress,controlProgressEnd).map(request->_))
          val controlSample=if active.nonEmpty then controlInput(sample) else sample
          def combine(values: Vector[ControlNetResiduals]): Option[ControlNetResiduals]=
            if values.isEmpty then None
            else
              def sum(tensors: Vector[Tensor[Float]]): Tensor[Float]=
                val first=tensors.head
                require(tensors.forall(t=>t.shape==first.shape && t.device==first.device && t.dtype==first.dtype),
                  "ControlNet residual shapes differ between units")
                val output=new Array[Float](first.shape.numel.toInt)
                tensors.foreach { tensor =>
                  val data=tensor.toFloatArray
                  var i=0
                  while i<output.length do {output(i)=(output(i)+data(i)).toFloat;i+=1}
                }
                Tensor.fromFloats(output,first.shape)
              Some(ControlNetResiduals(values.head.down.indices.map(index=>sum(values.map(_.down(index)))).toVector,
                sum(values.map(_.middle))))
          val baseCond=conditionalOverride.getOrElse(conditional)
          val basePosAdded=positiveAddedOverride.orElse(positiveAdded)
          val baseUncond=unconditionalOverride.orElse(unconditional)
          val baseNegAdded=negativeAddedOverride.orElse(negativeAdded)
          val positiveControls=combine(active.map { case (request,scales) =>
            request.model(controlSample,timesteps,baseCond,request.image,scales)
          })
          val negativeControls=if baseUncond.isEmpty then None else combine(active.collect {
            case (request,scales) if !request.conditionalOnly =>
              request.model(controlSample,timesteps,baseUncond.get,request.image,scales)
          })
          def input(isPositive: Boolean): (Tensor[Float],Tensor[Float]) =
            val applyDetail=detail.filter(d=>baseUncond.isEmpty || d.mode=="both" || (isPositive && d.mode=="cond") || (!isPositive && d.mode=="uncond"))
            applyDetail.fold((sample,timesteps))(d=>(Tensor.fromFloats(sample.toFloatArray.map(_*d.scale),sample.shape),
              Tensor.fromFloats(Array(d.timestep),Shape.from(Seq(1L)))))
          val (effCond,effUncond)=cads.fold((baseCond,baseUncond)) { case (state,prog) =>
            state.step(baseCond,baseUncond,prog)
          }
          def evalCpuModel(sTensor: Tensor[Float],tTensor: Tensor[Float],cTensor: Tensor[Float],addedOpt: Option[DiffusionAddedConditioning],ctrlOpt: Option[ControlNetResiduals]): Variable =
            multiDiffusion match
              case None => model(Variable(sTensor),tTensor,Variable(cTensor),addedOpt,ctrlOpt)
              case Some(mdCfg) =>
                val th=math.max(8,mdCfg.tileHeight/8);val tw=math.max(8,mdCfg.tileWidth/8);val ov=math.max(0,mdCfg.tileOverlap/8)
                val h=sTensor.shape.dims(2).toInt;val w=sTensor.shape.dims(3).toInt
                if h<=th && w<=tw then model(Variable(sTensor),tTensor,Variable(cTensor),addedOpt,ctrlOpt)
                else
                  val tiles=MultiDiffusion.tileGrid(h,w,th,tw,ov)
                  val gauss=mdCfg.method.toLowerCase(java.util.Locale.ROOT).startsWith("mixture")
                  val outC=model.convOut.weight.data.shape.dims.head.toInt
                  val batch=sTensor.shape.dims(0).toInt
                  val plane=h*w
                  val acc=new Array[Float](batch*outC*plane)
                  val wsum=new Array[Float](plane)
                  tiles.foreach { case (y0,y1,x0,x1) =>
                    val tileS=MultiDiffusion.sliceSpatial(sTensor,y0,y1,x0,x1)
                    val tileCtrl=ctrlOpt.map(c => MultiDiffusion.sliceResiduals(c,h,y0,y1,x0,x1))
                    val tileOut=model(Variable(tileS),tTensor,Variable(cTensor),addedOpt,tileCtrl).data.toFloatArray
                    val tileH=y1-y0;val tileW=x1-x0;val tilePlane=tileH*tileW
                    val wt=if gauss then MultiDiffusion.gaussianWeights(tileH,tileW) else Array.fill(tilePlane)(1f)
                    for ty <- 0 until tileH; tx <- 0 until tileW do
                      val wVal=wt(ty*tileW+tx)
                      val dstIdx=(y0+ty)*w+(x0+tx)
                      val srcIdx=ty*tileW+tx
                      wsum(dstIdx)+=wVal
                      for b <- 0 until batch; ch <- 0 until outC do
                        acc((b*outC+ch)*plane+dstIdx)+=tileOut((b*outC+ch)*tilePlane+srcIdx)*wVal
                  }
                  for b <- 0 until batch; ch <- 0 until outC; p <- 0 until plane do
                    acc((b*outC+ch)*plane+p)/=math.max(wsum(p),1e-12f)
                  Variable(Tensor.fromFloats(acc,Shape.from(Seq(batch.toLong,outC.toLong,h.toLong,w.toLong))),requiresGrad=false)
          val (positiveSampleBase,positiveTime)=input(true)
          refStep.foreach { case (bank,refSample) =>
            bank.bank.clear()
            bank.resetLayerCounter(ReferenceOnly.Mode.Write)
            ReferenceOnly.withBank(Some(bank)) {
              evalCpuModel(refSample,positiveTime,effCond,basePosAdded,None)
            }
          }
          val chgShifted=if effUncond.isDefined && guidance>1f then
            chgState.flatMap { case (st,stepIdx) =>
              st.stepCorrection(stepIdx,timesteps.toFloatArray(0),guidance,sample.shape,(uIn,cIn) => {
                val uOut=NegPip.withSigns(negativeSigns)(evalCpuModel(uIn,positiveTime,effUncond.get,baseNegAdded,negativeControls)).data
                val cOut=NegPip.withSigns(positiveSigns)(evalCpuModel(cIn,positiveTime,effCond,basePosAdded,positiveControls)).data
                (uOut,cOut)
              },sample,prediction)
            }
          else None
          val positiveSample=chgShifted.fold(positiveSampleBase)(_._2)
          val sagStore=sagStep.map(_ => new Sag.MapStore())
          val positive=Sag.withStore(sagStore) {
            refStep.foreach(_._1.resetLayerCounter(ReferenceOnly.Mode.ReadCond))
            ReferenceOnly.withBank(refStep.map(_._1)) {
              NegPip.withSigns(positiveSigns)(evalCpuModel(positiveSample,positiveTime,effCond,basePosAdded,positiveControls))
            }
          }
          val pagDiff=pag.filter(_.scale>0).map { pCfg =>
            val pScale=pCfg.effectiveScale(timesteps.toFloatArray(0))
            if pScale<=0f then None
            else
              val perturbed=NegPip.withSigns(positiveSigns)(Pag.withPerturbed(pCfg.layers)(evalCpuModel(positiveSample,positiveTime,effCond,basePosAdded,positiveControls)))
              val posArr=positive.data.toFloatArray;val pertArr=perturbed.data.toFloatArray
              Some(Array.tabulate(posArr.length)(idx => pScale*(posArr(idx)-pertArr(idx))))
          }.flatten
          def addPagAndSag(baseOut: Tensor[Float]): Tensor[Float] =
            val withPag=pagDiff.fold(baseOut) { diff =>
              val arr=baseOut.toFloatArray
              Tensor.fromFloats(Array.tabulate(arr.length)(idx => arr(idx)+diff(idx)),baseOut.shape)
            }
            sagStep.filter(_._1.scale!=0.0).fold(withPag) { case (sagCfg,effSigma) =>
              val degSample=Sag.degradeLatent(positiveSample,positive.data,effSigma,sagStore.flatMap(_.midAttentionProbs),sagCfg.blurSigma)
              val degCtx=effUncond.getOrElse(effCond)
              val degAdded=if effUncond.isDefined then baseNegAdded else basePosAdded
              val degCtrl=if effUncond.isDefined then negativeControls else positiveControls
              val degOut=evalCpuModel(degSample,positiveTime,degCtx,degAdded,degCtrl).data
              Sag.applyGuidance(withPag,positive.data,degOut,sagCfg.scale)
            }
          VaeExecution.checkCancelled(cancelled)
          addPagAndSag(effUncond.fold(positive.data) { context =>
            val (negativeSampleBase,negativeTime)=input(false)
            val negativeSample=chgShifted.fold(negativeSampleBase)(_._1)
            val negativeWithRef=Incantations.withSeg(segUncond) {
              refStep.foreach(_._1.resetLayerCounter(ReferenceOnly.Mode.ReadUncond))
              ReferenceOnly.withBank(refStep.map(_._1)) {
                NegPip.withSigns(negativeSigns)(evalCpuModel(negativeSample,negativeTime,context,baseNegAdded,negativeControls))
              }
            }
            val negative=refStep.filter(_._1.config.styleFidelity>0.0).fold(negativeWithRef) { case (bank,_) =>
              val plainNeg=Incantations.withSeg(segUncond)(NegPip.withSigns(negativeSigns)(evalCpuModel(negativeSample,negativeTime,context,baseNegAdded,negativeControls)))
              Variable(ReferenceOnly.blendUncond(negativeWithRef.data,plainNeg.data,bank.config.styleFidelity),requiresGrad=false)
            }
            VaeExecution.checkCancelled(cancelled)
            val synthPositive=neutralPlan.fold(positive) { np =>
              val negArr=negative.data.toFloatArray
              val deltas=np.branches.zipWithIndex.map { case (br,bIdx) =>
                val bOut=if bIdx==0 && br.loraScope.isEmpty then positive.data
                else scalatorch.nn.LinearAdapterScope.withMultipliers(br.loraScope) {
                  NegPip.withSigns(br.posSigns)(evalCpuModel(positiveSample,positiveTime,br.cond,br.posAdded,positiveControls)).data
                }
                val bArr=bOut.toFloatArray
                Tensor.fromFloats(Array.tabulate(bArr.length)(idx => bArr(idx)-negArr(idx)),bOut.shape)
              }
              val comb=NeutralPrompt.combineDeltas(np.tree,deltas).toFloatArray
              Variable(Tensor.fromFloats(Array.tabulate(comb.length)(idx => negArr(idx)+comb(idx)),negative.data.shape),requiresGrad=false)
            }
            val ghPositive=guidanceHook.filter(_=>guidance>1f).fold(synthPositive)(gh =>
              Variable(gh.rewriteConditional(sample,negative.data,synthPositive.data,timesteps.toFloatArray(0),guidance,prediction)))
            val effectivePositive=dynamicThresholding.filter(_=>guidance>1f).fold(ghPositive)(dt =>
              Variable(dt.rewriteConditional(sample,negative.data,ghPositive.data,timesteps.toFloatArray(0),guidance,prediction)))
            val guidedOut=guided(negative,effectivePositive,guidance,rescale,scheduled)
            neutralPlan.filter(_.phi>0.0).fold(guidedOut)(np => NeutralPrompt.rescale(guidedOut,synthPositive.data,np.phi))
          })
          } }
        }
      )

  final case class Cuda(deviceIndex: Int=0,maxBytes: Long=1073741824L,override val offloadWeights: Boolean=false) extends UNetExecution:
    require(deviceIndex>=0 && maxBytes>0,"invalid CUDA U-Net device or memory budget")
    val device=s"cuda:$deviceIndex"
    private[models] def withPredictor[A](model: ConditionalUNet2d,conditional: Tensor[Float],
        unconditional: Option[Tensor[Float]],cancelled: () => Boolean,
        positiveAdded: Option[DiffusionAddedConditioning]=None,negativeAdded: Option[DiffusionAddedConditioning]=None,
        controlNets: Vector[DiffusionControlNetRequest]=Vector.empty)(body: UNetPredictor => A): A =
      VaeExecution.checkCancelled(cancelled)
      require(conditional.shape.rank==3 && unconditional.forall(_.shape.rank==3),"U-Net contexts must be rank three")
      require(unconditional.isDefined || negativeAdded.isEmpty,"negative addition requires negative context")
      val positiveFeatures=model.addedFeatures(conditional.shape.dims.head,positiveAdded)
      val negativeFeatures=unconditional.flatMap(c => model.addedFeatures(c.shape.dims.head,negativeAdded))
      CudaSession.withDevice(deviceIndex,maxBytes) { session =>
        Using.Manager { use =>
          val unet=use(CudaUNet.upload(session,model,cancelled,offloadWeights))
          val controls=controlNets.map { request =>
            val runtime=use(CudaDiffusionRuntime.upload(session,request.model,cancelled,
              offloadWeights || controlNets.size>1))
            val image=use(session.upload(request.model.orderedImage(request.image)))
            (request,runtime,image)
          }
          val positiveContext=use(session.upload(conditional))
          val negativeContext=unconditional.map(x => use(session.upload(x)))
          val positiveAddition=positiveFeatures.map(x => use(session.upload(x)))
          val negativeAddition=negativeFeatures.map(x => use(session.upload(x)))
          // [negative x N; positive x N] (positive x N without guidance) for batched sampling, built once per denoising loop.
          var repeated: Option[(Int,Boolean,scalatorch.ffm.CudaTensor,Option[scalatorch.ffm.CudaTensor])]=None
          def repeatedConditioning(count: Int,pair: Boolean): (scalatorch.ffm.CudaTensor,Option[scalatorch.ffm.CudaTensor]) =
            repeated match
              case Some((n,p,context,addition)) if n==count && p==pair => (context,addition)
              case _ =>
                // The manager also owns these handles; CudaTensor.close is idempotent.
                repeated.foreach { case (_,_,context,addition) => context.close();addition.foreach(_.close()) }
                repeated=None
                def tile(negative: Option[scalatorch.ffm.CudaTensor],positive: scalatorch.ffm.CudaTensor): scalatorch.ffm.CudaTensor =
                  use(session.cat((if pair then Seq.fill(count)(negative.get) else Seq.empty)++Seq.fill(count)(positive),0))
                val context=tile(negativeContext,positiveContext)
                val addition=positiveAddition.map(p => tile(negativeAddition,p))
                repeated=Some((count,pair,context,addition))
                (context,addition)
          body(new UNetPredictor:
            def apply(sample: Tensor[Float],timesteps: Tensor[Float],guidance: Float,
                controlProgress: Double,rescale: Float,scheduled: Option[Float],detail: Option[DetailDaemon.Adjustment],controlProgressEnd: Double,
                dynamicThresholding: Option[DynamicThresholdingConfig],prediction: DiffusionPrediction,cads: Option[(CadsState,Double)],
                guidanceHook: Option[GuidanceHookState],pag: Option[PagConfig],
                conditionalOverride: Option[Tensor[Float]],positiveAddedOverride: Option[DiffusionAddedConditioning],
                unconditionalOverride: Option[Tensor[Float]],negativeAddedOverride: Option[DiffusionAddedConditioning],
                positiveSigns: Option[Array[Float]],negativeSigns: Option[Array[Float]],
                kohyaHrFix: Option[KohyaHrFixConfig],tokenDownsampling: Option[(TokenDownsamplingConfig,Boolean,Int,Int)],
                multiDiffusion: Option[MultiDiffusionConfig],segUncond: Option[(Double,Int,Int)],
                chgState: Option[(CharacteristicGuidance.State,Int)],
                neutralPlan: Option[NeutralPrompt.StepPlan],
                sagStep: Option[(SagConfig,Double)],
                refStep: Option[(ReferenceOnly.BankState,Tensor[Float])]): Tensor[Float] =
              VaeExecution.checkCancelled(cancelled)
              require(guidance.isFinite && guidance>=0,"invalid guidance scale")
              // Plain classifier-free guidance: both passes go through the U-Net as one batch of two. A sample batch of N that shares
              // one prompt ("batched sampling") runs as one batch of 2N, [negative x N; positive x N], or of N without guidance.
              def batchedPlain(): Option[Tensor[Float]] =
                val count=if sample.shape.rank==4 then sample.shape.dims.head.toInt else 0
                val multi=count>1 && conditional.shape.dims.head==1L && unconditional.forall(_.shape.dims.head==1L)
                val pair=negativeContext.isDefined && guidance>1f
                val plain=sample.shape.rank==4 && controls.isEmpty &&
                  detail.isEmpty && cads.isEmpty && refStep.isEmpty && chgState.isEmpty && multiDiffusion.isEmpty && pag.isEmpty && sagStep.isEmpty &&
                  neutralPlan.isEmpty && segUncond.isEmpty && positiveSigns.isEmpty && negativeSigns.isEmpty && conditionalOverride.isEmpty &&
                  unconditionalOverride.isEmpty && positiveAddedOverride.isEmpty && negativeAddedOverride.isEmpty && kohyaHrFix.isEmpty &&
                  tokenDownsampling.isEmpty && guidanceHook.isEmpty && dynamicThresholding.isEmpty && scheduled.isEmpty && rescale==0f &&
                  (!pair || positiveAddition.isDefined==negativeAddition.isDefined) && CudaUNet.batchedGuidance
                val eligible=plain && (if multi then pair || negativeContext.isEmpty else count==1 && pair)
                if !eligible then
                  // One context cannot serve N samples on the general path: the caller samples the images one by one instead.
                  if multi then throw new BatchedSamplingUnsupported("this step cannot run as one U-Net batch")
                  None
                else
                  try
                    Using.Manager { temporary =>
                      val input=temporary(session.upload(sample))
                      val (stacked,context,addition)=
                        if multi then
                          val (c,a)=repeatedConditioning(count,pair)
                          (if pair then temporary(session.cat(Seq(input,input),0)) else input,c,a)
                        else (temporary(session.cat(Seq(input,input),0)),
                          temporary(session.cat(Seq(negativeContext.get,positiveContext),0)),
                          positiveAddition.map(p => temporary(session.cat(Seq(negativeAddition.get,p),0))))
                      val output=temporary(unet.applyPrepared(stacked,timesteps,context,addition,None))
                      VaeExecution.checkCancelled(cancelled)
                      val both=output.toCpu
                      if !pair then Some(both)
                      else
                        val values=both.toFloatArray
                        val half=values.length/2
                        // same operation order as the two-pass path: negative+(positive+(-negative))*guidance
                        val guided=Array.tabulate(half) { i =>
                          val n=values(i);val p=values(half+i)
                          n+(p+(-n))*guidance
                        }
                        Some(Tensor.fromFloats(guided,Shape.from(Seq(count.toLong)++both.shape.dims.drop(1))))
                    }.get
                  catch
                    // A single image falls back to two passes; a batch goes back to the caller, which samples the images one by one.
                    case error: Throwable if !multi && CudaUNet.memoryFailure(error) => None
              batchedPlain().getOrElse(
              KohyaHrFix.withStep(kohyaHrFix,timesteps.toFloatArray(0)) {
              val tdStep=tokenDownsampling.getOrElse((TokenDownsamplingConfig(),false,0,1))
              TokenDownsampling.withStep(tokenDownsampling.map(_._1),tdStep._2,tdStep._3,tdStep._4,model.downBlocks.size) {
              Using.Manager { temporary =>
                val input=temporary(session.upload(sample))
                val controlSample=if controls.nonEmpty && sample.shape.dims(1)!=4 then
                  temporary(session.upload(controlInput(sample))) else input
                def residuals(context: scalatorch.ffm.CudaTensor,includeConditionalOnly: Boolean,addition: Option[scalatorch.ffm.CudaTensor]=None): Option[CudaControlNetResiduals]=
                  var combined: Option[CudaControlNetResiduals]=None
                  controls.foreach { case (request,runtime,image) =>
                    if includeConditionalOnly || !request.conditionalOnly then
                      request.scales(controlProgress,controlProgressEnd).foreach { values =>
                        val contribution=temporary(runtime.controlNet(controlSample,request.model,timesteps,context,image,values,addition))
                        combined match
                          case None => combined=Some(contribution)
                          case Some(previous) =>
                            val down=previous.down.indices.map(i=>temporary(session.add(previous.down(i),contribution.down(i)))).toVector
                            val middle=temporary(session.add(previous.middle,contribution.middle))
                            // The manager also owns these handles; CudaTensor.close is idempotent.
                            previous.close()
                            contribution.close()
                            combined=Some(CudaControlNetResiduals(down,middle))
                      }
                  }
                  combined
                val baseCond=conditionalOverride.getOrElse(conditional)
                val basePosCtx=conditionalOverride.fold(positiveContext)(c => temporary(session.upload(c)))
                val basePosAddition=positiveAddedOverride.fold(positiveAddition)(pa => model.addedFeatures(baseCond.shape.dims.head,Some(pa)).map(x => temporary(session.upload(x))))
                val baseUncond=unconditionalOverride.orElse(unconditional)
                val baseNegCtx=unconditionalOverride.fold(negativeContext)(u => Some(temporary(session.upload(u))))
                val baseNegAddition=negativeAddedOverride.fold(negativeAddition)(na => baseUncond.flatMap(u => model.addedFeatures(u.shape.dims.head,Some(na)).map(x => temporary(session.upload(x)))))
                val positiveControl=residuals(basePosCtx,true,basePosAddition)
                def adjusted(isPositive: Boolean): (scalatorch.ffm.CudaTensor,Tensor[Float]) =
                  val selected=detail.filter(d=>baseNegCtx.isEmpty || d.mode=="both" || (isPositive && d.mode=="cond") || (!isPositive && d.mode=="uncond"))
                  selected.fold((input,timesteps))(d=>(temporary(session.scale(input,d.scale)),Tensor.fromFloats(Array(d.timestep),Shape.from(Seq(1L)))))
                val (effPosCtx,effNegCtx)=cads match
                  case Some((state,prog)) =>
                    val (cCpu,uCpu)=state.step(baseCond,baseUncond,prog)
                    val pGpu=if cCpu eq baseCond then basePosCtx else temporary(session.upload(cCpu))
                    val nGpu=uCpu.map(u => if baseUncond.exists(_ eq u) then baseNegCtx.get else temporary(session.upload(u)))
                    (pGpu,nGpu)
                  case None => (basePosCtx,baseNegCtx)
                def evalCudaModel(sGpu: scalatorch.ffm.CudaTensor,tTensor: Tensor[Float],cGpu: scalatorch.ffm.CudaTensor,addGpu: Option[scalatorch.ffm.CudaTensor],ctrlGpu: Option[CudaControlNetResiduals]): scalatorch.ffm.CudaTensor =
                  multiDiffusion match
                    case None => temporary(unet.applyPrepared(sGpu,tTensor,cGpu,addGpu,ctrlGpu))
                    case Some(mdCfg) =>
                      val th=math.max(8,mdCfg.tileHeight/8);val tw=math.max(8,mdCfg.tileWidth/8);val ov=math.max(0,mdCfg.tileOverlap/8)
                      val h=sGpu.shape.dims(2).toInt;val w=sGpu.shape.dims(3).toInt
                      if h<=th && w<=tw then temporary(unet.applyPrepared(sGpu,tTensor,cGpu,addGpu,ctrlGpu))
                      else
                        val sCpu=sGpu.toCpu
                        val ctrlCpu=ctrlGpu.map(c => ControlNetResiduals(c.down.map(_.toCpu),c.middle.toCpu))
                        val tiles=MultiDiffusion.tileGrid(h,w,th,tw,ov)
                        val gauss=mdCfg.method.toLowerCase(java.util.Locale.ROOT).startsWith("mixture")
                        val outC=model.convOut.weight.data.shape.dims.head.toInt
                        val batch=sCpu.shape.dims(0).toInt
                        val plane=h*w
                        val acc=new Array[Float](batch*outC*plane)
                        val wsum=new Array[Float](plane)
                        tiles.foreach { case (y0,y1,x0,x1) =>
                          val tileS=temporary(session.upload(MultiDiffusion.sliceSpatial(sCpu,y0,y1,x0,x1)))
                          val tileCtrl=ctrlCpu.map { c =>
                            val sliced=MultiDiffusion.sliceResiduals(c,h,y0,y1,x0,x1)
                            CudaControlNetResiduals(sliced.down.map(d => temporary(session.upload(d))),temporary(session.upload(sliced.middle)))
                          }
                          val tileOutGpu=temporary(unet.applyPrepared(tileS,tTensor,cGpu,addGpu,tileCtrl))
                          val tileOut=tileOutGpu.toCpu.toFloatArray
                          val tileH=y1-y0;val tileW=x1-x0;val tilePlane=tileH*tileW
                          val wt=if gauss then MultiDiffusion.gaussianWeights(tileH,tileW) else Array.fill(tilePlane)(1f)
                          for ty <- 0 until tileH; tx <- 0 until tileW do
                            val wVal=wt(ty*tileW+tx)
                            val dstIdx=(y0+ty)*w+(x0+tx)
                            val srcIdx=ty*tileW+tx
                            wsum(dstIdx)+=wVal
                            for b <- 0 until batch; ch <- 0 until outC do
                              acc((b*outC+ch)*plane+dstIdx)+=tileOut((b*outC+ch)*tilePlane+srcIdx)*wVal
                        }
                        for b <- 0 until batch; ch <- 0 until outC; p <- 0 until plane do
                          acc((b*outC+ch)*plane+p)/=math.max(wsum(p),1e-12f)
                        temporary(session.upload(Tensor.fromFloats(acc,Shape.from(Seq(batch.toLong,outC.toLong,h.toLong,w.toLong)))))
                val (positiveInputBase,positiveTime)=adjusted(true)
                refStep.foreach { case (bank,refSample) =>
                  bank.bank.clear()
                  bank.resetLayerCounter(ReferenceOnly.Mode.Write)
                  val refGpu=temporary(session.upload(refSample))
                  ReferenceOnly.withBank(Some(bank)) {
                    evalCudaModel(refGpu,positiveTime,effPosCtx,basePosAddition,None)
                  }
                }
                val chgShifted=if effNegCtx.isDefined && guidance>1f then
                  chgState.flatMap { case (st,stepIdx) =>
                    st.stepCorrection(stepIdx,timesteps.toFloatArray(0),guidance,sample.shape,(uIn,cIn) => {
                      val uGpu=temporary(session.upload(uIn))
                      val cGpu=temporary(session.upload(cIn))
                      val uOut=NegPip.withSigns(negativeSigns)(evalCudaModel(uGpu,positiveTime,effNegCtx.get,baseNegAddition,residuals(effNegCtx.get,false,baseNegAddition))).toCpu
                      val cOut=NegPip.withSigns(positiveSigns)(evalCudaModel(cGpu,positiveTime,effPosCtx,basePosAddition,positiveControl)).toCpu
                      (uOut,cOut)
                    },sample,prediction)
                  }
                else None
                val positiveInput=chgShifted.fold(positiveInputBase)(s => temporary(session.upload(s._2)))
                val sagStore=sagStep.map(_ => new Sag.MapStore())
                val positive=Sag.withStore(sagStore) {
                  refStep.foreach(_._1.resetLayerCounter(ReferenceOnly.Mode.ReadCond))
                  ReferenceOnly.withBank(refStep.map(_._1)) {
                    NegPip.withSigns(positiveSigns)(evalCudaModel(positiveInput,positiveTime,effPosCtx,basePosAddition,positiveControl))
                  }
                }
                val pagDiff=pag.filter(_.scale>0).map { pCfg =>
                  val pScale=pCfg.effectiveScale(timesteps.toFloatArray(0))
                  if pScale<=0f then None
                  else
                    val perturbed=NegPip.withSigns(positiveSigns)(Pag.withPerturbed(pCfg.layers)(evalCudaModel(positiveInput,positiveTime,effPosCtx,basePosAddition,positiveControl)))
                    val posArr=positive.toCpu.toFloatArray;val pertArr=perturbed.toCpu.toFloatArray
                    Some(Array.tabulate(posArr.length)(idx => pScale*(posArr(idx)-pertArr(idx))))
                }.flatten
                val sagOutOpt=sagStep.filter(_._1.scale!=0.0).map { case (sagCfg,effSigma) =>
                  val posCpuForSag=positive.toCpu
                  val degSample=Sag.degradeLatent(positiveInput.toCpu,posCpuForSag,effSigma,sagStore.flatMap(_.midAttentionProbs),sagCfg.blurSigma)
                  val degGpu=temporary(session.upload(degSample))
                  val degCtx=effNegCtx.getOrElse(effPosCtx)
                  val degAdd=if effNegCtx.isDefined then baseNegAddition else basePosAddition
                  val degCtrl=if effNegCtx.isDefined then residuals(degCtx,false,degAdd) else positiveControl
                  val degOut=evalCudaModel(degGpu,positiveTime,degCtx,degAdd,degCtrl).toCpu
                  (posCpuForSag,degOut,sagCfg.scale)
                }
                def addPagAndSag(baseOut: Tensor[Float]): Tensor[Float] =
                  val withPag=pagDiff.fold(baseOut) { diff =>
                    val arr=baseOut.toFloatArray
                    Tensor.fromFloats(Array.tabulate(arr.length)(idx => arr(idx)+diff(idx)),baseOut.shape)
                  }
                  sagOutOpt.fold(withPag) { case (posCpuForSag,degOut,sagScale) =>
                    Sag.applyGuidance(withPag,posCpuForSag,degOut,sagScale)
                  }
                val positiveCpu=if (rescale!=0 || scheduled.nonEmpty || dynamicThresholding.nonEmpty || guidanceHook.nonEmpty || neutralPlan.nonEmpty) && effNegCtx.nonEmpty then Some(positive.toCpu) else None
                var scheduledResult: Option[Tensor[Float]]=None
                VaeExecution.checkCancelled(cancelled)
                val predicted=effNegCtx.fold(positive) { context =>
                  val negativeControl=residuals(context,false,baseNegAddition)
                  val (negativeInputBase,negativeTime)=adjusted(false)
                  val negativeInput=chgShifted.fold(negativeInputBase)(s => temporary(session.upload(s._1)))
                  val negativeWithRef=Incantations.withSeg(segUncond) {
                    refStep.foreach(_._1.resetLayerCounter(ReferenceOnly.Mode.ReadUncond))
                    ReferenceOnly.withBank(refStep.map(_._1)) {
                      NegPip.withSigns(negativeSigns)(evalCudaModel(negativeInput,negativeTime,context,baseNegAddition,negativeControl))
                    }
                  }
                  val negative=refStep.filter(_._1.config.styleFidelity>0.0).fold(negativeWithRef) { case (bank,_) =>
                    val plainNeg=Incantations.withSeg(segUncond)(NegPip.withSigns(negativeSigns)(evalCudaModel(negativeInput,negativeTime,context,baseNegAddition,negativeControl)))
                    temporary(session.upload(ReferenceOnly.blendUncond(negativeWithRef.toCpu,plainNeg.toCpu,bank.config.styleFidelity)))
                  }
                  VaeExecution.checkCancelled(cancelled)
                  if scheduled.nonEmpty || dynamicThresholding.nonEmpty || guidanceHook.nonEmpty || neutralPlan.nonEmpty then
                    val negCpu=negative.toCpu
                    val synthPosCpu=neutralPlan.fold(positiveCpu.get) { np =>
                      val negArr=negCpu.toFloatArray
                      val deltas=np.branches.zipWithIndex.map { case (br,bIdx) =>
                        val bOut=if bIdx==0 then positiveCpu.get
                        else
                          val bCtx=temporary(session.upload(br.cond))
                          val bAdd=br.posAdded.fold(basePosAddition)(pa => model.addedFeatures(br.cond.shape.dims.head,Some(pa)).map(x => temporary(session.upload(x))))
                          NegPip.withSigns(br.posSigns)(evalCudaModel(positiveInput,positiveTime,bCtx,bAdd,positiveControl)).toCpu
                        val bArr=bOut.toFloatArray
                        Tensor.fromFloats(Array.tabulate(bArr.length)(idx => bArr(idx)-negArr(idx)),bOut.shape)
                      }
                      val comb=NeutralPrompt.combineDeltas(np.tree,deltas).toFloatArray
                      Tensor.fromFloats(Array.tabulate(comb.length)(idx => negArr(idx)+comb(idx)),negCpu.shape)
                    }
                    val ghPos=guidanceHook.filter(_=>guidance>1f).fold(synthPosCpu)(gh =>
                      gh.rewriteConditional(sample,negCpu,synthPosCpu,timesteps.toFloatArray(0),guidance,prediction))
                    val effPos=dynamicThresholding.filter(_=>guidance>1f).fold(ghPos)(dt =>
                      dt.rewriteConditional(sample,negCpu,ghPos,timesteps.toFloatArray(0),guidance,prediction))
                    val gOut=Autograd.noGrad { guided(Variable(negCpu),Variable(effPos),guidance,rescale,scheduled) }
                    scheduledResult=Some(neutralPlan.filter(_.phi>0.0).fold(gOut)(np => NeutralPrompt.rescale(gOut,synthPosCpu,np.phi)))
                  // Preserve CPU subtraction and CFG operation ordering and rounding.
                  val negated=temporary(session.scale(negative,-1f))
                  val difference=temporary(session.add(positive,negated));negated.close();positive.close()
                  val scaled=temporary(session.scale(difference,guidance));difference.close()
                  val result=temporary(session.add(negative,scaled));negative.close();scaled.close()
                  result
                }
                input.close()
                VaeExecution.checkCancelled(cancelled)
                val result=predicted.toCpu
                VaeExecution.checkCancelled(cancelled)
                addPagAndSag(scheduledResult.getOrElse(positiveCpu.fold(result)(positive=>GuidanceRescale(result,positive,rescale))))
              }.get } })
          )
        }.get
      }

  def fromEnvironment(environment: Map[String,String]): UNetExecution =
    val device=environment.getOrElse("SDFX_SCALATORCH_UNET_DEVICE","cpu")
    def cuda(index: Int)=Cuda(index,environment.get("SDFX_SCALATORCH_UNET_MAX_BYTES").fold(1073741824L)(scalatorch.ffm.CudaSession.parseBudget),
      VaeExecution.offloadFromEnvironment(environment,"SDFX_SCALATORCH_UNET_OFFLOAD_WEIGHTS"))
    device match
      case "cpu" => Cpu
      case "cuda" => cuda(0)
      case value if value.matches("cuda:[0-9]+") => cuda(value.drop(5).toInt)
      case _ => throw IllegalArgumentException("U-Net device must be cpu, cuda or cuda:<index>")
