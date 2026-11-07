package scalatorch.models.worker

import scalatorch.models.{ClipFastTokenizer,ClipTextualInversion}
import scalatorch.Tensor
import NativeJson.*

/** Python-free operations supported by the embedding inspector. */
private[worker] object NativeEmbeddingInspector:
  def prepareMix(params: Map[String,NativeJson],streams: Vector[(String,ClipFastTokenizer,Tensor[Float])],
      context: NativeWorkContext): NativeEmbeddingMixer.Vectors =
    applyPreparedMix(params,prepareMixInputs(params,streams,context))

  def applyPreparedMix(params: Map[String,NativeJson],prepared: Vector[(NativeEmbeddingMixer.Vectors,Double)]): NativeEmbeddingMixer.Vectors =
    def bool(name: String): Boolean=params.get(name) match
      case None | Some(Bool(false)) => false
      case Some(Bool(true)) => true
      case _ => throw IllegalArgumentException(s"$name must be boolean")
    val expression=params.get("eval") match
      case None | Some(Str("")) => None
      case Some(Str(value)) if value.trim.isEmpty => None
      case Some(Str(value)) => Some(value)
      case _ => throw IllegalArgumentException("eval must be text")
    if bool("batch_presets") then throw new NativeBackendError("unsupported","batch presets are not supported by the native mixer")
    NativeEmbeddingMixer.mix(prepared,bool("concat"),bool("combine"),expression)

  def prepareMixInputs(params: Map[String,NativeJson],streams: Vector[(String,ClipFastTokenizer,Tensor[Float])],
      context: NativeWorkContext): Vector[(NativeEmbeddingMixer.Vectors,Double)] =
    val rows=params.get("rows") match
      case Some(Arr(values)) if values.nonEmpty && values.size<=128 => values
      case _ => throw IllegalArgumentException("choose 1 to 128 mixer rows")
    def vectors(result: Map[String,NativeJson]): NativeEmbeddingMixer.Vectors =
      result("encoders").asInstanceOf[Obj].fields.map { (key,value) =>
        val rows=value.asInstanceOf[Obj].fields("vectors").asInstanceOf[Arr].values.map { row =>
          row.asInstanceOf[Arr].values.map { case Number(n) => n.toFloat;case _ => throw IllegalStateException("invalid vector") }
        }
        key->rows
      }
    rows.flatMap {
      case Obj(row) =>
        context.checkCancelled()
        val weight=row.get("weight") match
          case None => 1.0
          case Some(Number(value)) if value.toDouble.isFinite => value.toDouble
          case _ => throw IllegalArgumentException("weight must be finite")
        val name=row.get("name") match
          case None => ""
          case Some(Str(value)) => value.trim
          case _ => throw IllegalArgumentException("invalid mixer name")
        if weight==0 || (name.isEmpty && !row.contains("ids")) then None
        else
          val values=row.get("ids") match
            case Some(Obj(ids)) =>
              require(ids.keySet==streams.map(_._1).toSet,"token IDs must be supplied for every encoder")
              streams.map { stream =>
                val id=ids(stream._1) match
                  case Number(value) if value.isValidLong && value.toLong>=0 => value.toLong
                  case _ => throw IllegalArgumentException("invalid token ID")
                val request=params.updated("text",Str(s"#$id"))
                stream._1->vectors(inspect(request,Vector(stream),context,includeNearest=false))(stream._1)
              }.toMap
            case Some(_) => throw IllegalArgumentException("invalid token IDs")
            case None =>
              require(!(streams.size==2 && name.startsWith("#")),"supply token IDs for each encoder on SDXL")
              vectors(inspect(params.updated("text",Str(name)),streams,context,includeNearest=false))
          Some(values->weight)
      case _ => throw IllegalArgumentException("invalid mixer row")
    }

  def exportVectors(params: Map[String,NativeJson],streams: Vector[(String,ClipFastTokenizer,Tensor[Float])],
      context: NativeWorkContext): Map[String,NativeJson] =
    val output=params.get("output") match
      case Some(Str(value)) if value.nonEmpty && value.toLowerCase(java.util.Locale.ROOT).endsWith(".txt") => java.nio.file.Path.of(value)
      case _ => throw IllegalArgumentException("vector export requires a .txt path")
    val overwrite=params.get("overwrite") match
      case None | Some(Bool(false)) => false
      case Some(Bool(true)) => true
      case _ => throw IllegalArgumentException("overwrite must be boolean")
    val inspected=inspect(params,streams,context,includeNearest=false)
    val encoders=inspected("encoders").asInstanceOf[Obj].fields.map { (key,value) =>
      key->value.asInstanceOf[Obj].fields("vectors")
    }
    val target=output.toAbsolutePath.normalize()
    val parent=target.getParent
    require(parent!=null && java.nio.file.Files.isDirectory(parent),"vector export directory does not exist")
    context.checkCancelled()
    val temp=java.nio.file.Files.createTempFile(parent,".embedding-vector-",".tmp")
    try
      java.nio.file.Files.writeString(temp,NativeJson.render(Obj(encoders)),java.nio.charset.StandardCharsets.UTF_8)
      context.checkCancelled()
      context.beginPublication()
      if overwrite then java.nio.file.Files.move(temp,target,java.nio.file.StandardCopyOption.ATOMIC_MOVE,
        java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      else java.nio.file.Files.createLink(target,temp)
      Map("path"->Str(output.toString))
    finally java.nio.file.Files.deleteIfExists(temp)

  final case class LoadedEmbedding(name: String,path: java.nio.file.Path,count: Int,leftWidth: Int,rightWidth: Option[Int])
  def list(params: Map[String,NativeJson],loaded: Vector[LoadedEmbedding],context: NativeWorkContext): Map[String,NativeJson] =
    val paths=params.get("embedding_paths") match
      case None => Vector.empty
      case Some(Arr(values)) if values.size<=100000 => values.map {
        case Str(value) => java.nio.file.Path.of(value)
        case _ => throw IllegalArgumentException("invalid embedding paths")
      }
      case _ => throw IllegalArgumentException("invalid embedding paths")
    val names=paths.flatMap { path =>
      context.checkCancelled()
      val filename=Option(path.getFileName).map(_.toString).getOrElse("")
      val dot=filename.lastIndexOf('.')
      if dot<0 || !Set(".safetensors",".pt",".bin")(filename.substring(dot).toLowerCase(java.util.Locale.ROOT)) then None
      else Some(filename.substring(0,dot).toLowerCase(java.util.Locale.ROOT))
    }.distinct.sorted
    val active=loaded.map(_.name.toLowerCase(java.util.Locale.ROOT)).toSet
    val entries=loaded.sortBy(_.name.toLowerCase(java.util.Locale.ROOT)).map { item =>
      context.checkCancelled()
      val encoders=Map("clip_l"->Arr(Vector(Number(item.count),Number(item.leftWidth))))++
        item.rightWidth.map(width=>"clip_g"->Arr(Vector(Number(item.count),Number(width))))
      Obj(Map("name"->Str(item.name),"path"->Str(item.path.toString),"encoders"->Obj(encoders),
        "file_metadata_status"->Str("file changed or could not be verified")))
    }
    Map("loaded"->Arr(entries),"available"->Arr(names.filterNot(active).map(Str(_))))

  def inspect(params: Map[String,NativeJson],streams: Vector[(String,ClipFastTokenizer,Tensor[Float])],
      context: NativeWorkContext,includeNearest: Boolean=true): Map[String,NativeJson] =
    val name=params.get("text") match
      case Some(Str(value)) if value.trim.nonEmpty => value.trim
      case _ => throw IllegalArgumentException("token is empty")
    val limit=params.get("limit") match
      case None => 30
      case Some(Number(n)) if n.isValidInt && n.toInt>=1 && n.toInt<=1000 => n.toInt
      case _ => throw IllegalArgumentException("invalid result limit")
    val paths=params.get("embedding_paths") match
      case None => Vector.empty
      case Some(Arr(values)) if values.size<=100000 => values.map {
        case Str(path) => java.nio.file.Path.of(path)
        case _ => throw IllegalArgumentException("invalid embedding paths")
      }
      case _ => throw IllegalArgumentException("invalid embedding paths")
    def num(value: Double): NativeJson =
      require(value.isFinite,"embedding contains non-finite values")
      Number(BigDecimal(value))
    val active=streams.exists { (_,tokenizer,_) =>
      val ids=tokenizer.encode(name.toLowerCase(java.util.Locale.ROOT),addSpecialTokens=false)
      ids.nonEmpty && tokenizer.isAddedToken(ids.head) && tokenizer.inspectionPiece(ids.head)._1.equalsIgnoreCase(name)
    }
    val candidates=if active || name.startsWith("#") then Vector.empty else paths.filter { path =>
      val file=Option(path.getFileName).map(_.toString).getOrElse("")
      file.replaceFirst("\\.[^.]*$","").equalsIgnoreCase(name) &&
        Set(".safetensors",".pt",".bin").exists(file.toLowerCase(java.util.Locale.ROOT).endsWith)
    }
    require(candidates.size<=1,s"duplicate embedding name: $name")
    val fileData=candidates.headOption.map { path =>
      val left=streams.find(_._1=="clip_l").getOrElse(throw IllegalArgumentException("missing CLIP-L"))
      val right=streams.find(_._1=="clip_g")
      val width=left._3.shape.dims(1).toInt;val widthG=right.map(_._3.shape.dims(1).toInt).getOrElse(0)
      val lower=path.toString.toLowerCase(java.util.Locale.ROOT)
      val values=if lower.endsWith(".safetensors") then ClipTextualInversion.readSafetensors(path,right.nonEmpty,width,widthG)
        else ClipTextualInversion.readPyTorch(path,right.nonEmpty,width,widthG)
      val metadata=if lower.endsWith(".safetensors") then
        val archive=scalatorch.io.SafeTensors.open(path)
        try archive.metadata finally archive.close()
      else Map.empty[String,String]
      (values,metadata.updated("path",path.toString))
    }
    var loaded=false
    val result=streams.map { (key,tokenizer,tensor) =>
      context.checkCancelled()
      val shape=tensor.shape.dims
      require(shape.size==2 && shape(0)<=Int.MaxValue && shape(1)>0 && shape(1)<=Int.MaxValue,"invalid token embedding shape")
      val count=shape(0).toInt;val width=shape(1).toInt
      val ids=if fileData.nonEmpty then Vector.empty else if name.startsWith("#") then
        Vector(name.drop(1).toLongOption.getOrElse(throw IllegalArgumentException("invalid token ID")))
      else tokenizer.encode(name.toLowerCase(java.util.Locale.ROOT),addSpecialTokens=false).toVector
      require(fileData.nonEmpty || (ids.nonEmpty && ids.forall(id=>id>=0 && id<count)),"token ID is outside the loaded encoder vocabulary")
      val isAdded=ids.nonEmpty && tokenizer.isAddedToken(ids.head) && tokenizer.inspectionPiece(ids.head)._1.equalsIgnoreCase(name)
      val selected=if isAdded then ids else ids.take(1)
      loaded ||= isAdded
      val weights=tensor.toFloatArray
      require(weights.forall(_.isFinite),"embedding contains non-finite values")
      val rows=fileData match
        case Some((value,_)) => if key=="clip_l" then value.clipL else value.clipG.getOrElse(throw IllegalArgumentException("missing CLIP-G"))
        case None => selected.map(id=>java.util.Arrays.copyOfRange(weights,id.toInt*width,(id.toInt+1)*width))
      val per=rows.map { row =>
        val norm=math.sqrt(row.iterator.map(x=>x.toDouble*x.toDouble).sum)
        val scores=if !includeNearest then Vector.empty else Vector.tabulate(count) { index =>
          if (index & 255)==0 then context.checkCancelled()
          var dot=0.0;var squared=0.0;var column=0
          while column<width do
            val value=weights(index*width+column).toDouble
            dot+=value*row(column).toDouble;squared+=value*value;column+=1
          val score=if norm==0 then 0.0 else dot/(math.max(norm,1e-6)*math.max(math.sqrt(squared),1e-6))
          (index,score)
        }.sortBy { (index,score) => (-score,index) }.take(limit)
        val nearest=scores.map { (index,score) =>
          Obj(Map("id"->Number(index),"token"->Str(tokenizer.inspectionPiece(index.toLong)._1),"score"->num(score)))
        }
        Obj(Map("magnitude"->num(norm),"min"->num(row.min.toDouble),"max"->num(row.max.toDouble),"nearest"->Arr(nearest)))
      }
      val flat=rows.flatten
      val first=per.head.asInstanceOf[Obj].fields
      val token=if fileData.nonEmpty then name else tokenizer.inspectionPiece(selected.head)._1
      key->Obj(Map("ids"->Arr(selected.map(id=>Number(BigDecimal(id)))),"token"->Str(token),
        "shape"->Arr(Vector(Number(rows.size),Number(width))),"vectors"->Arr(rows.map(row=>Arr(row.toVector.map(x=>num(x.toDouble))))),
        "magnitude"->Arr(per.map(_.asInstanceOf[Obj].fields("magnitude"))),"min"->num(flat.min.toDouble),"max"->num(flat.max.toDouble),
        "nearest"->first("nearest"),"per_vector"->Arr(per)))
    }.toMap
    val metadata=if loaded then Map("source"->Str("loaded model"),"file_metadata_status"->Str("file changed or could not be verified"))
      else fileData.map(_._2.map((key,value)=>key->Str(value))).getOrElse(Map.empty[String,NativeJson])
    Map("encoders"->Obj(result),"metadata"->Obj(metadata))

  def execute(params: Map[String,NativeJson],tokenizers: Vector[(String,ClipFastTokenizer)],
      context: NativeWorkContext): Map[String,NativeJson] =
    val operation=params.get("operation") match
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException("invalid embedding inspector operation")
    if operation!="tokenize" then
      throw new NativeBackendError("unsupported",s"native embedding inspector does not support $operation yet")
    val text=params.get("text") match
      case None => ""
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException("text must be a string")
    require(text.codePointCount(0,text.length)<=65536,"text is too long")
    val streams=tokenizers.map { (name,tokenizer) =>
      context.checkCancelled()
      val ids=tokenizer.encode(text.toLowerCase(java.util.Locale.ROOT),addSpecialTokens=false).toVector
      val tokens=ids.take(128).map { id =>
        context.checkCancelled()
        val (piece,_,_)=tokenizer.inspectionPiece(id)
        Obj(Map("id"->Number(BigDecimal(id)),"token"->Str(piece)))
      }
      (name,tokens,ids.size>128)
    }
    Map("encoders"->Obj(streams.map((name,tokens,_)=>name->Arr(tokens)).toMap),
      "warnings"->Arr(streams.collect { case (name,_,true) =>
        Str(s"$name: only the first 128 tokens are displayed and sent to the mixer")
      }))
