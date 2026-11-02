package scalatorch.models

import scalatorch.*
import java.nio.file.{Files,Path}
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.text.Normalizer
import java.util.Locale
import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

final case class ClipTokenBatch(ids: Tensor[Long],attentionMask: Tensor[Long])

/** Explicit Transformers CLIPConverter fast profile; does not perform slow-tokenizer ftfy repair. */
final class ClipFastTokenizer private (vocab: Map[String,Int],ranks: Map[(String,String),Int],val maxLength: Int):
  import ClipFastTokenizer.*
  val vocabSize: Int=vocab.size
  val bosId: Long=vocab(bos).toLong
  val eosId: Long=vocab(eos).toLong
  def encode(text: String,addSpecialTokens: Boolean=true): Array[Long] =
    require(text.length<=1048576 && TokenizerJson.validUnicode(text),"invalid or oversized CLIP text")
    val normalizedBuilder=new java.lang.StringBuilder()
    val clean=spaces.matcher(Normalizer.normalize(text,Normalizer.Form.NFC)).replaceAll(" ")
    // HF Lowercase expands each code point independently (no contextual final sigma).
    clean.codePoints().forEach(cp => normalizedBuilder.append(new String(Character.toChars(cp)).toLowerCase(Locale.ROOT)))
    val normalized=normalizedBuilder.toString
    val output=ArrayBuffer.empty[Long]
    if addSpecialTokens then output+=bosId
    def piece(word: String): Unit =
      val bytes=word.getBytes(UTF_8)
      require(bytes.length<=16384,"CLIP BPE piece exceeds 16384 bytes")
      val symbols=ArrayBuffer.from(bytes.indices.map { i =>
        val s=alphabet(bytes(i)&255)+(if i==bytes.length-1 then "</w>" else "")
        if vocab.contains(s) then s else eos
      })
      var merging=true
      while merging && symbols.size>1 do
        var best=Int.MaxValue
        var selected= -1
        var i=0
        while i+1<symbols.size do
          ranks.get((symbols(i),symbols(i+1))).foreach { rank =>
            if rank<best then { best=rank;selected=i }
          }
          i+=1
        if selected<0 then merging=false
        else
          symbols(selected)=symbols(selected)+symbols(selected+1)
          symbols.remove(selected+1)
      require(output.size.toLong+symbols.size<=1048576,"too many CLIP token IDs")
      symbols.foreach(s => output+=vocab(s).toLong)
    def ordinary(segment: String): Unit =
      val matcher=words.matcher(segment)
      while matcher.find() do piece(matcher.group())
    val matcher=special.matcher(normalized)
    var start=0
    while matcher.find() do
      ordinary(normalized.substring(start,matcher.start()))
      output+=vocab(matcher.group()).toLong
      start=matcher.end()
    ordinary(normalized.substring(start))
    if addSpecialTokens then output+=eosId
    output.toArray
  def batch(texts: Seq[String],truncate: Boolean=false,maxElements: Long=1048576L): ClipTokenBatch =
    val count=texts.size.toLong*maxLength
    require(maxElements>=0 && count<=maxElements && count<=Int.MaxValue,"CLIP batch exceeds allocation limit")
    val ids=Array.fill(count.toInt)(eosId)
    val mask=new Array[Long](count.toInt)
    texts.zipWithIndex.foreach { (text,row) =>
      val encoded=encode(text)
      require(truncate || encoded.length<=maxLength,"CLIP text exceeds context; request explicit truncation")
      val length=math.min(encoded.length,maxLength)
      Array.copy(encoded,0,ids,row*maxLength,length)
      ids(row*maxLength+length-1)=eosId
      java.util.Arrays.fill(mask,row*maxLength,row*maxLength+length,1L)
    }
    val shape=Shape.from(Seq(texts.size.toLong,maxLength.toLong))
    ClipTokenBatch(Tensor.fromLongs(ids,shape),Tensor.fromLongs(mask,shape))

object ClipFastTokenizer:
  private val bos="<|startoftext|>"
  private val eos="<|endoftext|>"
  private val spaces=Pattern.compile("\\s+",Pattern.UNICODE_CHARACTER_CLASS)
  private val words=Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",Pattern.UNICODE_CHARACTER_CLASS)
  private val special=Pattern.compile(Pattern.quote(bos)+"|"+Pattern.quote(eos))
  private val alphabet: Vector[String]=
    var next=256
    (0 to 255).map { b =>
      if (b>=33 && b<=126) || (b>=161 && b<=172) || b>=174 then b.toChar.toString
      else { val s=next.toChar.toString;next+=1;s }
    }.toVector
  private def read(path: Path): String=Using.resource(Files.newInputStream(path)) { stream =>
    val bytes=stream.readNBytes(TokenizerJson.maxBytes+1)
    require(bytes.length<=TokenizerJson.maxBytes,"CLIP tokenizer file exceeds32MiB")
    UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString
  }
  def load(vocabPath: Path,mergesPath: Path,maxLength: Int=77): ClipFastTokenizer =
    fromText(read(vocabPath),read(mergesPath),maxLength)
  def fromText(vocabJson: String,mergesText: String,maxLength: Int=77): ClipFastTokenizer =
    require(maxLength>=2 && maxLength<=16384,"invalid CLIP context length")
    val vocab=TokenizerJson.parse(vocabJson).obj.map((token,id) => token->id.int)
    require(vocab.nonEmpty && vocab.size<=1000000 && vocab.keys.forall(_.nonEmpty) &&
      vocab.values.toSet==(0 until vocab.size).toSet,"invalid CLIP vocabulary IDs")
    require(vocab.contains(bos) && vocab.contains(eos),"missing CLIP special tokens")
    require(mergesText.length<=TokenizerJson.maxBytes && TokenizerJson.validUnicode(mergesText),"invalid CLIP merges")
    val lines=mergesText.linesIterator.toVector
    require(lines.headOption.contains("#version: 0.2"),"unsupported CLIP merges header")
    val pairs=lines.drop(1).filter(_.nonEmpty).map { line =>
      val parts=line.split(" ",-1)
      require(parts.length==2 && parts.forall(_.nonEmpty),"invalid CLIP merge pair")
      val a=parts(0);val b=parts(1)
      require(vocab.contains(a) && vocab.contains(b) && vocab.contains(a+b),"CLIP merge references missing vocabulary")
      (a,b)
    }
    require(pairs.size<=1000000 && pairs.distinct.size==pairs.size,"duplicate or excessive CLIP merges")
    new ClipFastTokenizer(vocab,pairs.zipWithIndex.toMap,maxLength)
