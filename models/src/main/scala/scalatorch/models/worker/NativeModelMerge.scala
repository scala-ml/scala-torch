package scalatorch.models.worker

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import scala.util.Using
import scalatorch.{DType,Layout,Shape,Storage,Tensor}
import scalatorch.io.{SafeDType,SafeTensorPlan,SafeTensorWrite,SafeTensors,SafeTensorsWriter}
import NativeJson.*

/** NUI checkpoint merge. Archives are indexed once; only one output tensor is materialized at a time. */
object NativeModelMerge:
  private val floatTypes=Set(SafeDType.F16,SafeDType.BF16,SafeDType.F32,SafeDType.F64)
  private val inBlock="^model\\.diffusion_model\\.input_blocks\\.(\\d+)\\..*".r
  private val outBlock="^model\\.diffusion_model\\.output_blocks\\.(\\d+)\\..*".r
  private def block(key: String): Option[String] =
    if key.startsWith("first_stage_model.") then None
    else key match
      case inBlock(n) => Some(f"IN${n.toInt}%02d")
      case outBlock(n) => Some(f"OUT${n.toInt}%02d")
      case x if x.startsWith("model.diffusion_model.middle_block.") => Some("M00")
      case _ => Some("BASE")
  private def weights(text: String,sdxl: Boolean,base: Double): Map[String,Double] =
    val n=if sdxl then 9 else 12
    val names=Vector.tabulate(n)(i=>f"IN$i%02d")++Vector("M00")++Vector.tabulate(n)(i=>f"OUT$i%02d")
    val trimmed=text.trim
    val values=if trimmed.toLowerCase(java.util.Locale.ROOT).startsWith("preset:") then
      val preset=trimmed.drop(7).trim.toUpperCase(java.util.Locale.ROOT)
      Vector.tabulate(2*n+1) { i => preset match
        case "GRAD_V" => math.rint(math.abs(i-n).toDouble/n*10000)/10000
        case "GRAD_A" => math.rint((1-math.abs(i-n).toDouble/n)*10000)/10000
        case value if value.startsWith("FLAT_") => value.drop(5).toInt/100.0
        case "IN_ONLY" => if i<n then 1.0 else 0.0
        case "OUT_ONLY" => if i>n then 1.0 else 0.0
        case "MID_ONLY" => if i==n then 1.0 else 0.0
        case _ => throw IllegalArgumentException(s"unknown block preset '$preset'")
      }
    else trimmed.split("[,\\s]+").toVector.filter(_.nonEmpty).map(_.toDouble)
    require(values.size==names.size || values.size==names.size+1,"invalid block weight count")
    require(values.forall(_.isFinite),"block weights must be finite")
    val (baseValue,blockValues)=if values.size==names.size+1 then (values.head,values.tail) else (base,values)
    (names.zip(blockValues).toMap+ ("BASE"->baseValue))
  private def floats(file: SafeTensors,key: String): Array[Float] =
    file.tensors(key).dtype match
      // readFloat allocates fresh, contiguous storage owned only by this merge.
      case SafeDType.F16 | SafeDType.BF16 | SafeDType.F32 => file.readFloat(key).storage.data
      case SafeDType.F64 => file.readDouble(key).toDoubleArray.map(_.toFloat)
      case SafeDType.I64 => file.readLong(key).toLongArray.map(_.toFloat)
      case SafeDType.BOOL => file.readBoolean(key).toBooleanArray.map(if _ then 1f else 0f)
      case SafeDType.I32 =>
        val b=ByteBuffer.wrap(file.readRaw(key)).order(ByteOrder.LITTLE_ENDIAN)
        Array.fill(b.remaining()/4)(b.getInt().toFloat)
  private def typed(values: Array[Float],shape: Shape,dtype: SafeDType): SafeTensorWrite =
    // values is a freshly read or constructed private array, no caller retains it.
    val tensor=new Tensor(new Storage(values,DType.Float32),Layout.contiguous(shape,values.length))
    dtype match
      case SafeDType.F16 => SafeTensorWrite.F16(tensor)
      case SafeDType.BF16 => SafeTensorWrite.BF16(tensor)
      case SafeDType.F32 => SafeTensorWrite.F32(tensor)
      case _ => throw IllegalArgumentException("invalid merge precision")
  def execute(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    def str(k: String,d: String=""): String=p.get(k) match
      case None | Some(Null) => d
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$k must be a string")
    def number(k: String,d: Double): Double=p.get(k) match
      case None | Some(Null) => d
      case Some(Number(value)) if value.toDouble.isFinite => value.toDouble
      case _ => throw IllegalArgumentException(s"$k must be finite")
    val unknown=p.keySet--Set("model_a","model_b","model_c","mode","output","alpha","beta","block_weights","vae_source","precision","overwrite","output_dir")
    require(unknown.isEmpty,s"unknown merge parameters: ${unknown.toVector.sorted.mkString(", ")}")
    val mode=str("mode","Weight sum");require(Set("Weight sum","Add difference","Sum twice","Triple sum")(mode),"unknown merge mode")
    def input(key: String): Path =
      val name=str(key)
      require(name.nonEmpty && Files.isRegularFile(Path.of(name)),s"$key model does not exist")
      Path.of(name).toRealPath()
    val a=input("model_a");val b=input("model_b")
    val cName=str("model_c");val c=Option.when(mode!="Weight sum" && cName.nonEmpty)(input("model_c"))
    require(mode=="Weight sum" || c.nonEmpty,"merge mode requires model C")
    val outputName=str("output");require(outputName.nonEmpty && outputName.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors"),"output must be a .safetensors path")
    val output=Path.of(outputName).toAbsolutePath.normalize();require(output.getParent!=null && Files.isDirectory(output.getParent),"output directory missing")
    val inputs=Vector(a,b)++c.toVector
    require(inputs.forall(path=>Files.isRegularFile(path) && path.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors")),"inputs must be safetensors files")
    val canonicalTarget=output.getParent.toRealPath().resolve(output.getFileName).normalize()
    require(!inputs.exists(path=>path==canonicalTarget || (Files.exists(output) && Files.isSameFile(path,output))),"output would overwrite an input model")
    val overwrite=p.get("overwrite") match
      case None | Some(Bool(false)) => false
      case Some(Bool(true)) => true
      case _ => throw IllegalArgumentException("overwrite must be boolean")
    require(overwrite || !Files.exists(output),"output already exists")
    val alpha=number("alpha",.5);val beta=number("beta",0);val weightText=str("block_weights")
    val vaeSource=str("vae_source","merge");require(Set("merge","A","B")(vaeSource),"invalid VAE source")
    val precision=str("precision","fp16").toLowerCase(java.util.Locale.ROOT)
    val dtype=precision match
      case "fp16" => SafeDType.F16
      case "bf16" => SafeDType.BF16
      case "fp32" => SafeDType.F32
      case _ => throw IllegalArgumentException("invalid merge precision")
    Using.resource(SafeTensors.open(a)) { fa => Using.resource(SafeTensors.open(b)) { fb =>
      val fc=c.map(SafeTensors.open(_))
      try
        val keys=fa.tensors.keys.toVector.sorted
        val sdxl=keys.exists(_.startsWith("conditioner."))
        require(fb.tensors.keys.exists(_.startsWith("conditioner."))==sdxl,"model architectures differ")
        val blockWeights=Option.when(weightText.trim.nonEmpty)(weights(weightText,sdxl,alpha))
        val recipe=Obj(Map("type"->Str("sdfx-merge"),"mode"->Str(mode),"alpha"->Number(BigDecimal(alpha)),"beta"->Number(BigDecimal(beta)),
          "block_weights"->Option.when(weightText.nonEmpty)(Str(weightText)).getOrElse(Null),"vae_source"->Str(vaeSource),"precision"->Str(precision),
          "model_a"->Str(a.getFileName.toString),"model_b"->Str(b.getFileName.toString),"model_c"->Option.when(cName.nonEmpty)(Str(Path.of(cName).getFileName.toString)).getOrElse(Null)))
        var skipped=0
        var loaded=0
        def raw(file: SafeTensors,key: String): SafeTensorWrite =
          val info=file.tensors(key);SafeTensorWrite.Raw(info.dtype,info.shape,file.readRaw(key))
        val plans=keys.map { key =>
          val ai=fa.tensors(key);val bi=fb.tensors.get(key)
          val isFloat=floatTypes(ai.dtype)
          val dimsA=ai.shape.dims
          val compatible=bi.exists { info =>
            info.shape==ai.shape || (dimsA.size==4 && info.shape.dims.size==4 && dimsA(0)==info.shape.dims(0) &&
              dimsA(2)==info.shape.dims(2) && dimsA(3)==info.shape.dims(3))
          }
          val copyA= !isFloat || bi.isEmpty || (block(key).nonEmpty || vaeSource=="merge") && !compatible
          if bi.isEmpty || (isFloat && !compatible && (block(key).nonEmpty || vaeSource=="merge")) then skipped+=1
          val selected=if !copyA && block(key).isEmpty && vaeSource=="B" then fb else fa
          val outShape=if copyA then ai.shape else if selected eq fb then bi.get.shape else ai.shape
          val outType=if copyA then ai.dtype else dtype
          val plan=SafeTensorPlan(outType,outShape,()=>{
            context.checkCancelled()
            if loaded%64==0 then context.event("progress",Map("step"->Number(loaded),"total"->Number(keys.size),"pass"->Str("merge")))
            loaded+=1
            if copyA then raw(fa,key)
            else if block(key).isEmpty && vaeSource!="merge" then typed(floats(selected,key),outShape,dtype)
            else
              val x=floats(fa,key);val y0=floats(fb,key)
              val y=if ai.shape==bi.get.shape then y0 else
                val out=x.clone();val ad=ai.shape.dims;val bd=bi.get.shape.dims
                val area=(ad(2)*ad(3)).toInt;val ac=ad(1).toInt;val bc=bd(1).toInt;val common=math.min(ac,bc)*area
                for batch<-0 until ad(0).toInt do System.arraycopy(y0,batch*bc*area,out,batch*ac*area,common)
                out
              val z=fc.flatMap(file=>file.tensors.get(key).filter(_.shape==ai.shape).map(_=>floats(file,key)))
              val w=block(key).flatMap(name=>blockWeights.flatMap(_.get(name))).getOrElse(alpha).toFloat
              val bweight=beta.toFloat
              val zv=z.orNull
              var i=0
              while i<x.length do
                val a=x(i);val b=y(i)
                x(i)=mode match
                  case "Weight sum" => a*(1f-w)+b*w
                  case "Add difference" => if zv==null then a else a+(b-zv(i))*w
                  case "Sum twice" =>
                    val first=a*(1f-w)+b*w
                    if zv==null then first else first*(1f-bweight)+zv(i)*bweight
                  case _ => if zv==null then a*(1f-w)+b*w else a*(1f-w-bweight)+b*w+zv(i)*bweight
                i+=1
              typed(x,outShape,dtype)
          })
          key->plan
        }.toMap
        SafeTensorsWriter.writePlanned(output,plans,Map("sd_merge_recipe"->NativeJson.render(recipe),"format"->"pt"),
          overwrite=overwrite,cancelled=context.cancelled,beforePublish=()=>context.beginPublication())
        context.event("progress",Map("step"->Number(keys.size),"total"->Number(keys.size),"pass"->Str("merge")))
        Map("path"->Str(output.toString),"keys"->Number(keys.size),"skipped"->Number(skipped),"sdxl"->Bool(sdxl),"recipe"->recipe)
      finally fc.foreach(_.close())
    }}
