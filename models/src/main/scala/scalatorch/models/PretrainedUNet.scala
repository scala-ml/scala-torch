package scalatorch.models

import scalatorch.*
import scalatorch.io.SafeTensors
import scalatorch.nn.*
import java.nio.file.Path
import scala.collection.mutable
import scala.util.Using

/** Diffusers single-file safetensors loader. All parameters are frozen unless trainable is requested. */
object PretrainedUNet:
  def load(directory: Path,trainable: Boolean=false,maxParameters: Long=1000000000L): ConditionalUNet2d =
    require(maxParameters>=0,"negative U-Net parameter budget")
    val c=UNetConfig.read(directory.resolve("config.json"))
    Using.resource(SafeTensors.open(directory.resolve("diffusion_pytorch_model.safetensors"))) { archive =>
      val total=archive.tensors.values.foldLeft(BigInt(0))((sum,t) => sum+t.shape.numel)
      require(total<=maxParameters,"U-Net weights exceed parameter budget")
      val used=mutable.Set.empty[String]
      def parameter(name: String,dims: Int*): Parameter =
        val shape=Shape.from(dims.map(_.toLong))
        val info=archive.tensors.getOrElse(name,throw IllegalArgumentException(s"missing U-Net weight: $name"))
        require(info.shape==shape,s"U-Net weight shape mismatch: $name, expected $shape, got ${info.shape}")
        require(used.add(name),s"duplicate U-Net parameter use: $name")
        Parameter(archive.readFloat(name),trainable)
      def linear(name: String,in: Int,out: Int,bias: Boolean=true): Linear =
        Linear(parameter(s"$name.weight",out,in),if bias then Some(parameter(s"$name.bias",out)) else None)
      def conv(name: String,in: Int,out: Int,k: Int=3,stride: Int=1,padding: Int=1): Conv2d =
        Conv2d(parameter(s"$name.weight",out,in,k,k),Some(parameter(s"$name.bias",out)),
          Conv2dConfig(stride=(stride,stride),padding=(padding,padding)))
      def group(name: String,width: Int,epsilon: Double): GroupNorm =
        GroupNorm(c.groups,width,Some(parameter(s"$name.weight",width)),Some(parameter(s"$name.bias",width)),epsilon)
      def norm(name: String,width: Int): LayerNorm =
        LayerNorm(width,Some(parameter(s"$name.weight",width)),Some(parameter(s"$name.bias",width)),1e-5)
      def resnet(name: String,in: Int,out: Int,scale: Double=1): DiffusionResnet2d =
        DiffusionResnet2d(group(s"$name.norm1",in,c.epsilon),conv(s"$name.conv1",in,out),
          Some(linear(s"$name.time_emb_proj",c.timeWidth,out)),group(s"$name.norm2",out,c.epsilon),
          conv(s"$name.conv2",out,out),if in==out then None else Some(conv(s"$name.conv_shortcut",in,out,1,padding=0)),scale)
      def spatial(name: String,width: Int): DiffusionSpatialTransformer =
        def attention(prefix: String,context: Int): DiffusionAttention =
          DiffusionAttention(linear(s"$prefix.to_q",width,width,false),linear(s"$prefix.to_k",context,width,false),
            linear(s"$prefix.to_v",context,width,false),linear(s"$prefix.to_out.0",width,width),c.heads)
        val blocks=Vector.tabulate(c.transformerLayers) { i =>
          val p=s"$name.transformer_blocks.$i"
          DiffusionTransformerBlock(norm(s"$p.norm1",width),attention(s"$p.attn1",width),
            norm(s"$p.norm2",width),attention(s"$p.attn2",c.contextWidth),norm(s"$p.norm3",width),
            DiffusionFeedForward(linear(s"$p.ff.net.0.proj",width,width*8),linear(s"$p.ff.net.2",width*4,width)))
        }
        DiffusionSpatialTransformer(group(s"$name.norm",width,1e-6),conv(s"$name.proj_in",width,width,1,padding=0),
          blocks,conv(s"$name.proj_out",width,width,1,padding=0))
      def layer(prefix: String,index: Int,in: Int,out: Int,attention: Boolean): DiffusionUNetLayer =
        DiffusionUNetLayer(resnet(s"$prefix.resnets.$index",in,out),
          if attention then Some(spatial(s"$prefix.attentions.$index",out)) else None)
      val convIn=conv("conv_in",c.inChannels,c.channels.head)
      val time=DiffusionTimeEmbedding(linear("time_embedding.linear_1",c.channels.head,c.timeWidth),
        linear("time_embedding.linear_2",c.timeWidth,c.timeWidth))
      var current=c.channels.head
      val skips=mutable.ArrayBuffer(current)
      val down=c.channels.indices.toVector.map { i =>
        val out=c.channels(i);val prefix=s"down_blocks.$i"
        val layers=Vector.tabulate(c.layersPerBlock) { j =>
          val l=layer(prefix,j,current,out,c.downAttention(i))
          current=out;skips+=out;l
        }
        val sample=if i==c.channels.size-1 then None else
          skips+=out
          Some(DiffusionDownsample2d(conv(s"$prefix.downsamplers.0.conv",out,out,stride=2,padding=c.downsamplePadding)))
        DiffusionDownBlock(layers,sample)
      }
      val mid=DiffusionMidBlock(resnet("mid_block.resnets.0",current,current,c.midScale),
        spatial("mid_block.attentions.0",current),resnet("mid_block.resnets.1",current,current,c.midScale))
      val up=c.channels.reverse.zipWithIndex.map { (out,i) =>
        val prefix=s"up_blocks.$i"
        val layers=Vector.tabulate(c.layersPerBlock+1) { j =>
          val skip=skips.remove(skips.size-1)
          val l=layer(prefix,j,current+skip,out,c.upAttention(i))
          current=out;l
        }
        val sample=if i==c.channels.size-1 then None else
          Some(DiffusionUpsample2d(conv(s"$prefix.upsamplers.0.conv",out,out)))
        DiffusionUpBlock(layers,sample)
      }
      val model=ConditionalUNet2d(convIn,time,down,mid,up,group("conv_norm_out",current,c.epsilon),
        conv("conv_out",current,c.outChannels),DiffusionTimeSettings(c.flipSinToCos,c.frequencyShift))
      require(used.toSet==archive.tensors.keySet,s"unused U-Net weights: ${(archive.tensors.keySet--used).toVector.sorted.mkString(", ")}")
      model
    }
