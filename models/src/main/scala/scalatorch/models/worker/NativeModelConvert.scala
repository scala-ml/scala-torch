package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scala.util.Using
import scalatorch.{Shape,Tensor}
import scalatorch.io.{PyTorchArchiveLimits,PyTorchTensorArchive,SafeDType,SafeTensorPlan,SafeTensorWrite,SafeTensors,SafeTensorsWriter}
import NativeJson.*

object NativeModelConvert:
  private val floats=Set(SafeDType.F16,SafeDType.BF16,SafeDType.F32,SafeDType.F64)
  private val unet="model.diffusion_model."
  private val junk=Vector("optimizer","model_ema.","lr_scheduler","global_step","epoch","loss")
  def execute(p: Map[String,NativeJson],ctx: NativeWorkContext): Map[String,NativeJson] =
    require((p.keySet--Set("model","output","precision","ema","fix_clip","drop_junk","bake_vae","overwrite","output_dir")).isEmpty,"unknown conversion parameters")
    def str(k: String,default: String=""): String=p.get(k) match
      case None | Some(Null) => default
      case Some(Str(s)) => s
      case _ => throw IllegalArgumentException(s"$k must be a string")
    def bool(k: String,default: Boolean): Boolean=p.get(k) match
      case None | Some(Null) => default
      case Some(Bool(b)) => b
      case _ => throw IllegalArgumentException(s"$k must be boolean")
    def file(k: String): Path =
      val name=str(k)
      require(name.nonEmpty && name.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") && Files.isRegularFile(Path.of(name)),s"$k must be a safetensors file")
      Path.of(name).toRealPath()
    val source=file("model")
    val vaeName=str("bake_vae")
    val vaePath=Option.when(vaeName.nonEmpty) {
      val path=Path.of(vaeName)
      require(Files.isRegularFile(path),"bake_vae must be a file")
      path.toRealPath()
    }
    val output=Path.of(str("output")).toAbsolutePath.normalize()
    require(output.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") && output.getParent!=null && Files.isDirectory(output.getParent),"invalid output path")
    val realTarget=output.getParent.toRealPath().resolve(output.getFileName).normalize()
    require((Vector(source)++vaePath.toVector).forall(p=>p!=realTarget && (!Files.exists(output) || !Files.isSameFile(p,output))),"output would overwrite an input")
    val overwrite=bool("overwrite",false)
    require(overwrite || !Files.exists(output),"output already exists")
    val precision=str("precision","fp16").toLowerCase(java.util.Locale.ROOT)
    val target=precision match
      case "keep" => None
      case "fp16" => Some(SafeDType.F16)
      case "bf16" => Some(SafeDType.BF16)
      case "fp32" => Some(SafeDType.F32)
      case _ => throw IllegalArgumentException("invalid precision")
    val ema=str("ema","no-ema");require(Set("no-ema","ema-only","keep")(ema),"invalid EMA mode")
    val fix=bool("fix_clip",true);val drop=bool("drop_junk",true)
    Using.resource(SafeTensors.open(source)) { src =>
      val vae=vaePath.filter(_.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors")).map(SafeTensors.open(_))
      val torchVae=vaePath.filterNot(_.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors")).map { path =>
        import PyTorchTensorArchive.*
        val limits=PyTorchArchiveLimits(maxArchiveBytes=4L*1024*1024*1024,maxEntries=8192,
          maxEntryBytes=1024L*1024*1024,maxTotalUncompressedBytes=4L*1024*1024*1024,
          maxPickleBytes=32*1024*1024,maxPickleOps=4000000,maxTensors=8192,maxElements=128L*1024*1024)
        val value=PyTorchTensorArchive.read(path,scalatorch.models.PyTorchCompatibility.limits(limits)).value
        val entries=value match
          case DictValue(root) => root.get("state_dict") match
            case Some(DictValue(state)) => state
            case Some(_) => throw IllegalArgumentException("VAE state_dict must be a dictionary")
            case None => root
          case _ => throw IllegalArgumentException("VAE archive must be a dictionary")
        entries.collect { case (key,TensorValue(t)) if !key.startsWith("loss") && !key.startsWith("model_ema") =>
          key.stripPrefix("first_stage_model.")->t
        }
      }.getOrElse(Map.empty[String,PyTorchTensorArchive.StoredTensor])
      try
        val vaeKeys=vae.map(_.tensors.keysIterator.filterNot(k=>k.startsWith("loss") || k.startsWith("model_ema"))
          .map(k=>(k.stripPrefix("first_stage_model."),k)).toMap).getOrElse(Map.empty[String,String])
        val keys=src.tensors.keys.toVector.sorted
        var dropped=0;var used=0;var fixed=0;var processed=0
        val plans=keys.flatMap { key =>
          if key.startsWith("model_ema.") && ema!="keep" || drop && !key.startsWith("model_ema.") && junk.exists(key.startsWith) then
            dropped+=1;None
          else
            val emaKey=if ema=="ema-only" && key.startsWith(unet) then
              "model_ema."+("diffusion_model"+key.drop(unet.length)).replace(".","") else ""
            val selected=if emaKey.nonEmpty && src.tensors.contains(emaKey) then {used+=1;emaKey} else key
            val sourceInfo=src.tensors(selected)
            val position=fix && (key=="position_ids" || key.endsWith(".position_ids")) && !key.startsWith("model_ema.")
            if position then
              require(sourceInfo.shape.rank>0 && sourceInfo.shape.numel==sourceInfo.shape.dims.last,"invalid position_ids shape")
              fixed+=1
            val vaeKey=if !position && key.startsWith("first_stage_model.") then
              vaeKeys.get(key.stripPrefix("first_stage_model.")).filter(k=>vae.get.tensors(k).shape==sourceInfo.shape) else None
            val torchTensor=if !position && vaeKey.isEmpty && key.startsWith("first_stage_model.") then
              torchVae.get(key.stripPrefix("first_stage_model.")).filter(t=>t.shape==sourceInfo.shape.dims) else None
            val selectedFile=if vaeKey.nonEmpty then vae.get else src
            val selectedKey=vaeKey.getOrElse(selected)
            val info=selectedFile.tensors(selectedKey)
            val inputType=torchTensor.map(_.dtype match
              case PyTorchTensorArchive.StorageDType.Float16 => SafeDType.F16
              case PyTorchTensorArchive.StorageDType.BFloat16 => SafeDType.BF16
              case PyTorchTensorArchive.StorageDType.Float32 => SafeDType.F32
              case _ => throw IllegalArgumentException("floating-point tensor has an integer dtype")).getOrElse(info.dtype)
            val outType=if position then SafeDType.I64 else if floats(inputType) then target.getOrElse(inputType) else inputType
            Some(key->SafeTensorPlan(outType,info.shape,()=>{
              ctx.checkCancelled()
              if processed%64==0 then ctx.event("progress",Map("step"->Number(processed),"total"->Number(keys.size),"pass"->Str("convert")))
              processed+=1
              if position then SafeTensorWrite.I64(Tensor.fromLongs(Array.tabulate(info.shape.numel.toInt)(_.toLong),info.shape))
              else if torchTensor.nonEmpty then
                val tensor=Tensor.fromFloats(torchTensor.get.values,info.shape)
                outType match
                  case SafeDType.F16 => SafeTensorWrite.F16(tensor)
                  case SafeDType.BF16 => SafeTensorWrite.BF16(tensor)
                  case SafeDType.F32 => SafeTensorWrite.F32(tensor)
                  case _ => throw IllegalArgumentException("unsupported VAE dtype")
              else if outType==info.dtype then SafeTensorWrite.Raw(info.dtype,info.shape,selectedFile.readRaw(selectedKey))
              else
                val tensor=if info.dtype==SafeDType.F64 then Tensor.fromFloats(selectedFile.readDouble(selectedKey).toDoubleArray.map(_.toFloat),info.shape)
                  else selectedFile.readFloat(selectedKey)
                outType match
                  case SafeDType.F16 => SafeTensorWrite.F16(tensor)
                  case SafeDType.BF16 => SafeTensorWrite.BF16(tensor)
                  case SafeDType.F32 => SafeTensorWrite.F32(tensor)
                  case _ => throw IllegalArgumentException("unsupported conversion")
            }))
        }.toMap
        require(ema!="ema-only" || used>0,"ema-only: this checkpoint has no EMA weights")
        val recipe=Obj(Map("source"->Str(source.getFileName.toString),"precision"->Str(precision),"ema"->Str(ema),
          "fix_clip"->Bool(fix),"drop_junk"->Bool(drop),"bake_vae"->vaePath.map(p=>Str(p.getFileName.toString)).getOrElse(Null)))
        SafeTensorsWriter.writePlanned(output,plans,Map("format"->"pt","sdfx_convert"->NativeJson.render(recipe)),
          overwrite=overwrite,cancelled=ctx.cancelled,beforePublish=()=>ctx.beginPublication())
        ctx.event("progress",Map("step"->Number(keys.size),"total"->Number(keys.size),"pass"->Str("convert")))
        Map("path"->Str(output.toString),"keys"->Number(plans.size),"dropped"->Number(dropped),
          "ema_weights_used"->Number(used),"clip_fixed"->Number(fixed),"vae_baked"->Bool(vaeKeys.nonEmpty || torchVae.nonEmpty))
      finally vae.foreach(_.close())
    }
