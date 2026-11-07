package scalatorch.models.worker

import scalatorch.Tensor
import scalatorch.models.{ClipFastTokenizer,ClipTextualInversion}
import java.nio.file.{Files,Path}
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import NativeJson.*

private[worker] object NativeEmbeddingEditor:
  type Stream=(String,ClipFastTokenizer,Tensor[Float])
  private def number(value: Double): NativeJson =
    require(value.isFinite,"embedding contains non-finite values")
    Number(BigDecimal(value))
  def fingerprint(path: Path,context: NativeWorkContext): Vector[String] =
    val before=Files.readAttributes(path,classOf[BasicFileAttributes])
    require(before.isRegularFile && before.size()<=64L*1024*1024,"invalid embedding file")
    val digest=MessageDigest.getInstance("SHA-256")
    val input=Files.newInputStream(path)
    try
      val buffer=new Array[Byte](65536)
      var n=input.read(buffer)
      while n>=0 do
        context.checkCancelled()
        if n>0 then digest.update(buffer,0,n)
        n=input.read(buffer)
    finally input.close()
    val after=Files.readAttributes(path,classOf[BasicFileAttributes])
    require(before.size()==after.size() && before.lastModifiedTime()==after.lastModifiedTime() &&
      before.creationTime()==after.creationTime() && java.util.Objects.equals(before.fileKey(),after.fileKey()),
      "embedding changed while reading")
    Vector(before.size().toString,before.lastModifiedTime().toString,before.creationTime().toString,
      String.valueOf(before.fileKey()),java.util.HexFormat.of().formatHex(digest.digest()))

  def open(name: String,path: Path,modelToken: String,guidance: String,streams: Vector[Stream],
      context: NativeWorkContext): Map[String,NativeJson] =
    require(name!=null && name.trim.nonEmpty && path!=null && modelToken!=null && modelToken.nonEmpty && guidance!=null,
      "invalid editor request")
    val references=if guidance.trim.isEmpty then Vector.empty else guidance.split(",",-1).toVector.map { item =>
      val colon=item.lastIndexOf(':')
      require(colon>=0,"guidance needs a word and color")
      val word=item.substring(0,colon).trim;val color=item.substring(colon+1).trim
      require(word.nonEmpty && color.matches("#[0-9a-fA-F]{6}"),"guidance needs a word and #RRGGBB color")
      word->color.toUpperCase(java.util.Locale.ROOT)
    }
    val before=fingerprint(path,context)
    val left=streams.find(_._1=="clip_l").getOrElse(throw IllegalArgumentException("missing CLIP-L"))
    val right=streams.find(_._1=="clip_g")
    val sdxl=right.nonEmpty
    val width=left._3.shape.dims(1).toInt
    val widthG=right.map(_._3.shape.dims(1).toInt).getOrElse(0)
    val data=if path.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
      ClipTextualInversion.readSafetensors(path,sdxl,width,widthG)
    else ClipTextualInversion.readPyTorch(path,sdxl,width,widthG)
    val details=streams.map { (key,tokenizer,tensor) =>
      context.checkCancelled()
      val w=tensor.shape.dims(1).toInt;val count=tensor.shape.dims.head.toInt
      val ids=tokenizer.encode(name,addSpecialTokens=false).toVector
      require(ids.nonEmpty && tokenizer.isAddedToken(ids.head) && tokenizer.inspectionPiece(ids.head)._1.equalsIgnoreCase(name) &&
        ids.forall(id=>id>=0 && id<count),"loaded embedding ownership is unknown")
      val weights=tensor.toFloatArray
      val rows=ids.map(id=>java.util.Arrays.copyOfRange(weights,id.toInt*w,(id.toInt+1)*w).toVector)
      val disk=(if key=="clip_l" then data.clipL else data.clipG.getOrElse(throw IllegalArgumentException("missing CLIP-G"))).map(_.toVector)
      require(rows==disk,"embedding file differs from loaded model; reload before editing")
      val lows=Array.fill(w)(Float.PositiveInfinity);val highs=Array.fill(w)(Float.NegativeInfinity)
      val base=math.min(tokenizer.baseVocabSize,count)
      require(base>0,"tokenizer vocabulary is empty")
      for index <- 0 until base do
        if (index & 255)==0 then context.checkCancelled()
        var column=0
        while column<w do
          val value=weights(index*w+column)
          require(value.isFinite,"embedding contains non-finite values")
          lows(column)=math.min(lows(column),value);highs(column)=math.max(highs(column),value);column+=1
      (key,tokenizer,weights,w,rows,lows,highs)
    }
    val vectors=details.map { (key,_,_,_,rows,_,_) => key->Arr(rows.map(row=>Arr(row.map(value=>number(value.toDouble))))) }.toMap
    val ranges=details.map { (key,_,_,w,_,lows,highs) =>
      key->Arr(Vector.tabulate(w)(i=>Arr(Vector(number(lows(i).toDouble),number(highs(i).toDouble)))))
    }.toMap
    val markers=references.map { (text,color) =>
      val values=details.map { (key,tokenizer,weights,w,_,lows,highs) =>
        val ids=tokenizer.encode(text.toLowerCase(java.util.Locale.ROOT),false)
        require(ids.nonEmpty && ids.head>=0 && ids.head<weights.length/w,"guidance text has no tokens")
        val offset=ids.head.toInt*w
        val row=Vector.tabulate(w) { i =>
          val low=lows(i).toDouble;val high=highs(i).toDouble
          if low==high then Null else number(math.max(0.0,math.min(1.0,(weights(offset+i)-low)/(high-low))))
        }
        key->Arr(row)
      }.toMap
      Obj(Map("text"->Str(text),"color"->Str(color),"markers"->Obj(values)))
    }
    val after=fingerprint(path,context)
    require(before==after,"embedding changed while reading")
    Map("name"->Str(name),"path"->Str(path.toString),"model_token"->Str(modelToken),
      "file_identity"->Arr(after.map(Str(_))),"vectors"->Obj(vectors),"ranges"->Obj(ranges),"guidance"->Arr(markers))
