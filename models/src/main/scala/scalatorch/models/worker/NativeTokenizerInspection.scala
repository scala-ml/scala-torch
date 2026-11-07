package scalatorch.models.worker

import scalatorch.models.ClipFastTokenizer
import java.io.ByteArrayOutputStream
import java.nio.{ByteBuffer,CharBuffer}
import java.nio.charset.{CodingErrorAction,StandardCharsets}
import scala.collection.mutable.ArrayBuffer
import NativeJson.*

/** NUI token inspector over the exact active native tokenizer. */
private[worker] object NativeTokenizerInspection:
  private val utf8=StandardCharsets.UTF_8
  def execute(p: Map[String,NativeJson],tokenizers: Vector[(String,String,ClipFastTokenizer)],
      context: NativeWorkContext): Map[String,NativeJson] =
    def text(key: String,default: String): String=p.get(key) match
      case None => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    val input=text("text","")
    require(input.codePointCount(0,input.length)<=65536,"text must be at most 65536 characters")
    val mode=text("input_mode","text")
    require(Set("text","ids")(mode),"input_mode must be text or ids")
    val parsed=if mode=="ids" then
      require(input.trim.isEmpty || input.matches("\\s*\\d+\\s*(?:,\\s*\\d+\\s*)*"),"IDs must be comma-separated nonnegative integers")
      Some(if input.trim.isEmpty then Vector.empty else input.split(",").toVector.map(s =>
        s.trim.toLongOption.getOrElse(throw IllegalArgumentException("invalid integer ID"))))
    else None
    val streams=tokenizers.map { (name,source,tokenizer) =>
      context.checkCancelled()
      val ids=parsed.getOrElse(tokenizer.encode(input,addSpecialTokens=false).toVector)
      require(ids.size<=8192,"inspection is limited to 8192 tokens per stream")
      val groups=ArrayBuffer.empty[NativeJson]
      val pending=new ByteArrayOutputStream()
      val all=new ByteArrayOutputStream()
      val decodedParts=ArrayBuffer.empty[String]
      val pendingIds=ArrayBuffer.empty[Long]
      def flush(force: Boolean): Unit =
        if pendingIds.nonEmpty then
          val bytes=pending.toByteArray
          val decoder=utf8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
          val buffer=ByteBuffer.wrap(bytes)
          val status=decoder.decode(buffer,CharBuffer.allocate(bytes.length+1),false)
          if force || status.isError || !buffer.hasRemaining then
            groups+=Obj(Map("ids"->Arr(pendingIds.toVector.map(id=>Number(BigDecimal(id)))),"text"->Str(new String(bytes,utf8))))
            pending.reset();pendingIds.clear()
      val tokens=ids.zipWithIndex.map { (id,index) =>
        if (index & 127)==0 then context.checkCancelled()
        val (piece,special,bytes)=tokenizer.inspectionPiece(id)
        if special then flush(true)
        pending.write(bytes);pendingIds+=id
        if tokenizer.isAddedToken(id) then
          if all.size()>0 then
            val previous=new String(all.toByteArray,utf8).trim
            if previous.nonEmpty then decodedParts+=previous
            all.reset()
          decodedParts+=piece
        else all.write(bytes)
        flush(false)
        Obj(Map("index"->Number(index+1),"id"->Number(BigDecimal(id)),"piece"->Str(piece),
          "text"->Str(new String(bytes,utf8)),"bytes_hex"->Str(bytes.map(b=>f"${b&255}%02x").mkString(" ")),
          "special"->Bool(special)))
      }
      flush(true)
      if all.size()>0 then decodedParts+=new String(all.toByteArray,utf8).trim
      Obj(Map("name"->Str(name),"source"->Str(source),"count"->Number(ids.size),
        "ids"->Arr(ids.map(id=>Number(BigDecimal(id)))),"content_capacity"->Number(if tokenizer.maxLength>2 then tokenizer.maxLength-2 else 75),
        "bos_id"->Number(BigDecimal(tokenizer.bosId)),"eos_id"->Number(BigDecimal(tokenizer.eosId)),
        "pad_id"->Number(BigDecimal(tokenizer.padId)),"decoded"->Str(decodedParts.mkString(" ")),
        "tokens"->Arr(tokens),"groups"->Arr(groups.toVector)))
    }
    Map("input_mode"->Str(mode),"streams"->Arr(streams))
