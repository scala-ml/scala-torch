package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.io.SafeTensorSet
import scalatorch.nn.*
import java.nio.file.Path
import scala.collection.mutable
import scala.util.Using

final case class VaeModel(config: VaeConfig,encoder: VaeEncoder,decoder: VaeDecoder) derives Module:
  require(encoder.latentChannels==config.latentChannels && decoder.latentChannels==config.latentChannels &&
    encoder.convIn.weight.data.shape.dims(1)==config.inChannels && decoder.convOut.weight.data.shape.dims.head==config.outChannels,
    "VAE model input/latent/output channel mismatch")
  def encodeMoments(image: Variable): Variable=encoder(image)
  def decode(latent: Variable): Variable=decoder(latent)
  def encode(image: Variable): VaePosterior=VaePosterior(encodeMoments(image))
  private def validateLatents(latent: Variable): Unit =
    val dims=latent.data.shape.dims
    require(dims.size==4 && dims(1)==config.latentChannels && dims(2)>0 && dims(3)>0,
      "VAE diffusion latents require matching NCHW channels and positive spatial dimensions")
  /** Explicitly convert raw posterior samples or means to diffusion-space latents. */
  def toDiffusionLatents(latent: Variable): Variable =
    validateLatents(latent)
    latent*config.scalingFactor.toFloat
  /** Inverse scaling before decoding; raw decode remains unscaled. */
  def decodeDiffusionLatents(latent: Variable): Variable =
    validateLatents(latent)
    decode(latent*(1/config.scalingFactor).toFloat)

/** All encoder/decoder weights, materialized as CPU Float32; no posterior sampling or pipeline scaling. */
object PretrainedVae:
  def load(directory: Path,trainable: Boolean=false,maxParameters: Long=1000000000L,
      maxFeatureElements: Long=536870912L,maxScoreElements: Long=16777216L,
      configOverride: Option[VaeConfig]=None,filename: String="diffusion_pytorch_model.safetensors"): VaeModel =
    require(maxParameters>=0 && maxFeatureElements>=0 && maxScoreElements>=0,"negative VAE allocation limit")
    val c=configOverride.getOrElse(VaeConfig.read(directory.resolve("config.json")))
    Using.resource(SafeTensorSet.open(directory,filename=filename)) { archive =>
      loadFromView(SingleFileTensorView(archive,archive.tensors.keysIterator.map(name => name -> name).toVector),
        c,trainable,maxParameters,maxFeatureElements,maxScoreElements)
    }

  private[models] def loadSingleFile(archive: SafeTensorSet,trainable: Boolean=false,
      maxParameters: Long=1000000000L,maxFeatureElements: Long=536870912L,
      maxScoreElements: Long=16777216L): VaeModel =
    SingleFileSd1Config.inspect(archive)
    loadFromView(SingleFileTensorView.vae(archive),SingleFileSd1Config.vaeConfig,
      trainable,maxParameters,maxFeatureElements,maxScoreElements)

  private[models] def loadSd1View(archive: SingleFileTensorView,maxScoreElements: Long): VaeModel =
    loadFromView(archive,SingleFileSd1Config.vaeConfig,false,1000000000L,536870912L,maxScoreElements)

  /** Loads the SD2 VAE from a checkpoint whose archive remains owned by the caller. */
  private[models] def loadSd2SingleFile(archive: SafeTensorSet,trainable: Boolean=false,
      maxParameters: Long=1000000000L,maxFeatureElements: Long=536870912L,
      maxScoreElements: Long=16777216L): VaeModel =
    SingleFileSd2Config.inspect(archive)
    loadFromView(SingleFileTensorView.vae(archive),SingleFileSd1Config.vaeConfig,
      trainable,maxParameters,maxFeatureElements,maxScoreElements)

  private[models] def loadSdxlView(archive: SingleFileTensorView,maxScoreElements: Long): VaeModel =
    loadFromView(archive,SingleFileSdxlConfig.vaeConfig,false,1000000000L,536870912L,maxScoreElements)

  /** Loads the SDXL VAE from a checkpoint whose archive remains owned by the caller. */
  private[models] def loadSdxlSingleFile(archive: SafeTensorSet,trainable: Boolean=false,
      maxParameters: Long=1000000000L,maxFeatureElements: Long=536870912L,
      maxScoreElements: Long=16777216L): VaeModel =
    SingleFileSdxlConfig.inspect(archive)
    loadFromView(SingleFileTensorView.vae(archive),SingleFileSdxlConfig.vaeConfig,
      trainable,maxParameters,maxFeatureElements,maxScoreElements)

  private def loadFromView(archive: SingleFileTensorView,c: VaeConfig,trainable: Boolean,
      maxParameters: Long,maxFeatureElements: Long,maxScoreElements: Long): VaeModel =
    require(maxParameters>=0 && maxFeatureElements>=0 && maxScoreElements>=0,"negative VAE allocation limit")
    locally {
      val total=archive.tensors.values.foldLeft(BigInt(0))((n,t) => n+t.shape.numel)
      require(total<=maxParameters,"VAE weights exceed parameter budget")
      val used=mutable.Set.empty[String]
      def parameter(name: String,dims: Seq[Int],allowConvProjection: Boolean=false): Parameter =
        val shape=Shape.from(dims.map(_.toLong))
        val info=archive.tensors.getOrElse(name,throw IllegalArgumentException(s"missing VAE weight: $name"))
        val singleton=allowConvProjection && dims.size==2 && info.shape.dims==shape.dims++Vector(1L,1L)
        require(info.shape==shape || singleton,s"VAE weight shape mismatch: $name")
        require(used.add(name),s"duplicate VAE parameter use: $name")
        Parameter(archive.readFloat(name).reshape(shape),trainable)
      def conv(name: String,in: Int,out: Int,k: Int=3,stride: Int=1,padding: Int=1): Conv2d =
        Conv2d(parameter(s"$name.weight",Seq(out,in,k,k)),Some(parameter(s"$name.bias",Seq(out))),
          Conv2dConfig(stride=(stride,stride),padding=(padding,padding)))
      def group(name: String,width: Int): GroupNorm =
        GroupNorm(c.groups,width,Some(parameter(s"$name.weight",Seq(width))),Some(parameter(s"$name.bias",Seq(width))),1e-6)
      def resnet(name: String,in: Int,out: Int): DiffusionResnet2d =
        DiffusionResnet2d(group(s"$name.norm1",in),conv(s"$name.conv1",in,out),None,group(s"$name.norm2",out),
          conv(s"$name.conv2",out,out),if in==out then None else Some(conv(s"$name.conv_shortcut",in,out,1,padding=0)))
      def mid(name: String,width: Int): VaeMidBlock =
        val attention=if !c.midAttention then None else
          val prefix=s"$name.attentions.0"
          val keys=if archive.tensors.contains(s"$prefix.to_q.weight") then Vector("to_q","to_k","to_v","to_out.0")
            else Vector("query","key","value","proj_attn")
          def linear(key: String)=Linear(parameter(s"$prefix.$key.weight",Seq(width,width),true),
            Some(parameter(s"$prefix.$key.bias",Seq(width))))
          Some(VaeAttention2d(group(s"$prefix.group_norm",width),
            DiffusionAttention(linear(keys(0)),linear(keys(1)),linear(keys(2)),linear(keys(3)),1,maxScoreElements)))
        VaeMidBlock(resnet(s"$name.resnets.0",width,width),attention,resnet(s"$name.resnets.1",width,width))
      var current=c.channels.head
      val down=c.channels.zipWithIndex.map { (out,i) =>
        val prefix=s"encoder.down_blocks.$i"
        val layers=Vector.tabulate(c.layersPerBlock) { j =>
          val layer=resnet(s"$prefix.resnets.$j",current,out);current=out;layer
        }
        val sample=if i==c.channels.size-1 then None else
          Some(DiffusionDownsample2d(conv(s"$prefix.downsamplers.0.conv",out,out,stride=2,padding=0)))
        VaeDownBlock(layers,sample)
      }
      val moments=c.latentChannels*2
      val encoder=VaeEncoder(conv("encoder.conv_in",c.inChannels,c.channels.head),down,mid("encoder.mid_block",current),
        group("encoder.conv_norm_out",current),conv("encoder.conv_out",current,moments),
        if c.quantConv then Some(conv("quant_conv",moments,moments,1,padding=0)) else None,maxFeatureElements)
      current=c.channels.last
      val up=c.channels.reverse.zipWithIndex.map { (out,i) =>
        val prefix=s"decoder.up_blocks.$i"
        val layers=Vector.tabulate(c.layersPerBlock+1) { j =>
          val layer=resnet(s"$prefix.resnets.$j",current,out);current=out;layer
        }
        val sample=if i==c.channels.size-1 then None else Some(DiffusionUpsample2d(conv(s"$prefix.upsamplers.0.conv",out,out)))
        VaeUpBlock(layers,sample)
      }
      val decoder=VaeDecoder(conv("decoder.conv_in",c.latentChannels,c.channels.last),mid("decoder.mid_block",c.channels.last),up,
        group("decoder.conv_norm_out",current),conv("decoder.conv_out",current,c.outChannels),
        if c.postQuantConv then Some(conv("post_quant_conv",c.latentChannels,c.latentChannels,1,padding=0)) else None,maxFeatureElements)
      require(used.toSet==archive.tensors.keySet,s"unused VAE weights: ${(archive.tensors.keySet--used).toVector.sorted.mkString(", ")}")
      VaeModel(c,encoder,decoder)
    }
