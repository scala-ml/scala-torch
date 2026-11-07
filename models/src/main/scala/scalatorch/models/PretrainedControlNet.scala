package scalatorch.models

import scalatorch.*
import scalatorch.io.{PyTorchArchiveLimits,PyTorchTensorArchive,SafeTensorSet}
import scalatorch.nn.*
import java.nio.file.{Files,Path}
import scala.collection.mutable
import scala.util.Using
import java.util.concurrent.CancellationException

/** Strict loader for SD1 ControlNet directories and standard single-file safetensors. */
object PretrainedControlNet:
  def load(directory: Path,maxParameters: Long=500000000L,maxScoreElements: Long=16777216L,
      cancelled: () => Boolean=() => false): ConditionalControlNet2d =
    loadWithProgress(directory,maxParameters,maxScoreElements,cancelled,()=>())

  private[models] def loadWithProgress(directory: Path,maxParameters: Long,maxScoreElements: Long,
      cancelled: () => Boolean,onFirstTensorData: () => Unit): ConditionalControlNet2d =
    require(maxParameters>=0 && maxScoreElements>=0,"negative ControlNet parameter or attention budget")
    def checkCancelled(): Unit=if cancelled() || Thread.currentThread().isInterrupted then
      throw new CancellationException("ControlNet loading cancelled")
    checkCancelled()
    val single=Files.isRegularFile(directory)
    val config=if single then SingleFileControlNet.config else ControlNetConfig.read(directory.resolve("config.json"))
    val c=config.unet
    Using.Manager { use =>
      val name=directory.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
      val torch=single && Vector(".pt",".pth",".ckpt",".bin").exists(name.endsWith)
      val archive=if torch then
        val limits=PyTorchArchiveLimits(maxArchiveBytes=2L*1024*1024*1024,maxEntries=5000,
          maxEntryBytes=256L*1024*1024,maxTotalUncompressedBytes=2L*1024*1024*1024,
          maxPickleBytes=16*1024*1024,maxPickleOps=4000000,maxTensors=4096,
          maxElements=64000000L,maxTotalTensorElements=maxParameters)
        SingleFileControlNet.pytorchView(PyTorchTensorArchive.read(directory,PyTorchCompatibility.limits(limits),cancelled,onFirstTensorData).value)
      else
        val raw=use(if single then SafeTensorSet.openFile(directory) else SafeTensorSet.open(directory,filename="diffusion_pytorch_model.safetensors"))
        if single then SingleFileControlNet.view(raw) else
          SingleFileTensorView(raw,raw.tensors.keysIterator.map(name=>name->name).toVector)
      checkCancelled()
      val total=archive.tensors.values.foldLeft(BigInt(0))((sum,t)=>sum+t.shape.numel)
      require(total<=maxParameters,"ControlNet weights exceed parameter budget")
      val used=mutable.Set.empty[String]
      def parameter(name: String,dims: Int*): Parameter =
        checkCancelled()
        val shape=Shape.from(dims.map(_.toLong))
        val info=archive.tensors.getOrElse(name,throw IllegalArgumentException(s"missing ControlNet weight: $name"))
        require(info.shape==shape,s"ControlNet weight shape mismatch: $name, expected $shape, got ${info.shape}")
        require(used.add(name),s"duplicate ControlNet parameter use: $name")
        val result=Parameter(archive.readFloat(name),false)
        checkCancelled()
        result
      def linear(name: String,in: Int,out: Int,bias: Boolean=true): Linear =
        Linear(parameter(s"$name.weight",out,in),if bias then Some(parameter(s"$name.bias",out)) else None)
      def conv(name: String,in: Int,out: Int,k: Int=3,stride: Int=1,padding: Int=1): Conv2d =
        Conv2d(parameter(s"$name.weight",out,in,k,k),Some(parameter(s"$name.bias",out)),
          Conv2dConfig(stride=(stride,stride),padding=(padding,padding)))
      def group(name: String,width: Int): GroupNorm =
        GroupNorm(c.groups,width,Some(parameter(s"$name.weight",width)),Some(parameter(s"$name.bias",width)),c.epsilon)
      def norm(name: String,width: Int): LayerNorm =
        LayerNorm(width,Some(parameter(s"$name.weight",width)),Some(parameter(s"$name.bias",width)),1e-5)
      def resnet(name: String,in: Int,out: Int): DiffusionResnet2d =
        DiffusionResnet2d(group(s"$name.norm1",in),conv(s"$name.conv1",in,out),
          Some(linear(s"$name.time_emb_proj",c.timeWidth,out)),group(s"$name.norm2",out),
          conv(s"$name.conv2",out,out),if in==out then None else Some(conv(s"$name.conv_shortcut",in,out,1,padding=0)),c.midScale)
      def spatial(name: String,width: Int,stage: Int): DiffusionSpatialTransformer =
        def attention(prefix: String,context: Int): DiffusionAttention =
          DiffusionAttention(linear(s"$prefix.to_q",width,width,false),linear(s"$prefix.to_k",context,width,false),
            linear(s"$prefix.to_v",context,width,false),linear(s"$prefix.to_out.0",width,width),c.headsAt(stage),maxScoreElements)
        val blocks=Vector.tabulate(c.transformerLayersAt(stage)) { i =>
          val p=s"$name.transformer_blocks.$i"
          DiffusionTransformerBlock(norm(s"$p.norm1",width),attention(s"$p.attn1",width),
            norm(s"$p.norm2",width),attention(s"$p.attn2",c.contextWidth),norm(s"$p.norm3",width),
            DiffusionFeedForward(linear(s"$p.ff.net.0.proj",width,width*8),linear(s"$p.ff.net.2",width*4,width)))
        }
        def projection(prefix: String): Conv2d =
          if !c.linearProjection then conv(prefix,width,width,1,padding=0)
          else
            val flat=parameter(s"$prefix.weight",width,width)
            Conv2d(Parameter(flat.data.reshape(Shape.from(Seq(width.toLong,width.toLong,1L,1L))),false),
              Some(parameter(s"$prefix.bias",width)),Conv2dConfig())
        DiffusionSpatialTransformer(group(s"$name.norm",width),projection(s"$name.proj_in"),blocks,projection(s"$name.proj_out"))
      val convIn=conv("conv_in",c.inChannels,c.channels.head)
      val time=DiffusionTimeEmbedding(linear("time_embedding.linear_1",c.channels.head,c.timeWidth),
        linear("time_embedding.linear_2",c.timeWidth,c.timeWidth))
      var current=c.channels.head
      val down=c.channels.indices.toVector.map { i =>
        val out=c.channels(i);val prefix=s"down_blocks.$i"
        val layers=Vector.tabulate(c.layersPerBlock) { j =>
          val attention=if c.downAttention(i) then Some(spatial(s"$prefix.attentions.$j",out,i)) else None
          val layer=DiffusionUNetLayer(resnet(s"$prefix.resnets.$j",current,out),attention)
          current=out;layer
        }
        val downsample=if i==c.channels.size-1 then None else
          Some(DiffusionDownsample2d(conv(s"$prefix.downsamplers.0.conv",out,out,stride=2,padding=c.downsamplePadding)))
        DiffusionDownBlock(layers,downsample)
      }
      val mid=DiffusionMidBlock(resnet("mid_block.resnets.0",current,current),
        spatial("mid_block.attentions.0",current,c.channels.size-1),resnet("mid_block.resnets.1",current,current))
      val conditionWidths=config.conditioningEmbeddingChannels
      val conditionIn=conv("controlnet_cond_embedding.conv_in",config.conditioningChannels,conditionWidths.head)
      val conditionBlocks=conditionWidths.sliding(2).zipWithIndex.flatMap { (pair,index) =>
        Vector(conv(s"controlnet_cond_embedding.blocks.${index*2}",pair.head,pair.head),
          conv(s"controlnet_cond_embedding.blocks.${index*2+1}",pair.head,pair.last,stride=2))
      }.toVector
      val conditionOut=conv("controlnet_cond_embedding.conv_out",conditionWidths.last,c.channels.head)
      val skips=mutable.ArrayBuffer(c.channels.head)
      down.zipWithIndex.foreach { (block,index) =>
        block.layers.foreach(_=>skips+=c.channels(index))
        if block.downsample.isDefined then skips+=c.channels(index)
      }
      val downProjections=skips.zipWithIndex.map((width,index)=>conv(s"controlnet_down_blocks.$index",width,width,1,padding=0)).toVector
      val middleProjection=conv("controlnet_mid_block",current,current,1,padding=0)
      val model=ConditionalControlNet2d(config,convIn,time,down,mid,
        ControlNetConditioningEmbedding(conditionIn,conditionBlocks,conditionOut),downProjections,middleProjection,maxScoreElements)
      require(used.toSet==archive.tensors.keySet,
        s"unused ControlNet weights: ${(archive.tensors.keySet--used).toVector.sorted.mkString(", ")}")
      model
    }.get
