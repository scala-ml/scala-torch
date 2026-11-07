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
final class ClipFastTokenizer private (vocab: Map[String,Int],ranks: Map[(String,String),Int],val maxLength: Int,val padId: Long,
    addedTokenIds: Map[String,Long]=Map.empty,vectorGroups: Map[String,Vector[String]]=Map.empty,
    nuiFallbackNormalization: Boolean=false):
  import ClipFastTokenizer.*
  def vocabSize: Int=vocab.size+addedTokenIds.size
  def baseVocabSize: Int=vocab.size
  val bosId: Long=vocab(bos).toLong
  val eosId: Long=vocab(eos).toLong
  private lazy val inspectionVocab: Map[Long,String]=vocab.map((piece,id) => id.toLong->piece)++addedTokenIds.map((piece,id)=>id->piece)
  private lazy val addedIds: Set[Long]=addedTokenIds.values.toSet
  def isAddedToken(id: Long): Boolean=addedIds.contains(id)
  /** Remove a complete added group and return old row IDs in the new vocabulary order. */
  def withoutAddedTokenGroup(name: String): (ClipFastTokenizer,Vector[Long]) =
    require(name!=null,"invalid textual-inversion token")
    val key=name.toLowerCase(Locale.ROOT)
    val removed=vectorGroups.getOrElse(key,throw IllegalArgumentException(s"unknown textual-inversion token: $name")).toSet
    val remaining=addedTokenIds.toVector.filterNot((token,_)=>removed(token)).sortBy(_._2)
    val keep=(0 until vocab.size).iterator.map(_.toLong).toVector++remaining.map(_._2)
    val ids=remaining.zipWithIndex.map { case ((token,_),index) => token->(vocab.size.toLong+index) }.toMap
    (new ClipFastTokenizer(vocab,ranks,maxLength,padId,ids,vectorGroups-key,nuiFallbackNormalization),keep)
  /** Exact vocabulary piece and bytes used by token inspection, including dynamically added tokens. */
  def inspectionPiece(id: Long): (String,Boolean,Array[Byte]) =
    val piece=inspectionVocab.getOrElse(id,throw IllegalArgumentException(s"ID $id is not in tokenizer vocabulary"))
    val special=id==bosId || id==eosId || id==padId
    val raw=if special || addedTokenIds.contains(piece) then piece.getBytes(UTF_8)
      else
        val bytes=piece.stripSuffix("</w>").map(c => inverseAlphabet.getOrElse(c,
          throw IllegalArgumentException("invalid CLIP byte alphabet"))).toArray
        if piece.endsWith("</w>") then bytes :+ 32.toByte else bytes
    (piece,special,raw)
  private lazy val addedTokenPattern: Pattern =
    val tokens=(addedTokenIds.keys.toVector++Vector(bos,eos)).sortBy(token => (-token.length,token))
    Pattern.compile(tokens.map(Pattern.quote).mkString("(?:", "|", ")"),
      if nuiFallbackNormalization then Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE else 0)
  def withAddedTokens(tokens: Vector[(String,Int)]): ClipFastTokenizer =
    require(tokens.nonEmpty && tokens.size<=256,"textual inversion must add between 1 and 256 tokens")
    val nextIds=scala.collection.mutable.LinkedHashMap.from(addedTokenIds)
    val nextGroups=scala.collection.mutable.Map.from(vectorGroups)
    tokens.foreach { (raw,count) =>
      val token=raw.toLowerCase(Locale.ROOT)
      require(raw.nonEmpty && raw.length<=256 && !raw.exists(c => Character.isWhitespace(c) || Character.isISOControl(c)) &&
        count>=1 && count<=256,"invalid textual-inversion token or vector count")
      require(!vocab.contains(token) && !nextIds.contains(token),s"textual-inversion token already exists: $raw")
      val names=Vector.tabulate(count)(index => if index==0 then token else s"${token}_$index")
      require(names.distinct.size==names.size && names.forall(name => !vocab.contains(name) && !nextIds.contains(name)),
        s"textual-inversion continuation token already exists: $raw")
      names.foreach(name => nextIds(name)=vocab.size.toLong+nextIds.size)
      nextGroups(token)=names
    }
    new ClipFastTokenizer(vocab,ranks,maxLength,padId,nextIds.toMap,nextGroups.toMap,nuiFallbackNormalization)
  /** Transformers CLIPTokenizer fallback used when the optional ftfy package is absent. */
  def withNuiFallbackNormalization: ClipFastTokenizer =
    new ClipFastTokenizer(vocab,ranks,maxLength,padId,addedTokenIds,vectorGroups,true)
  def encode(text: String,addSpecialTokens: Boolean=true): Array[Long] =
    require(text.length<=1048576 && TokenizerJson.validUnicode(text),"invalid or oversized CLIP text")
    val normalizedBuilder=new java.lang.StringBuilder()
    val nfc=Normalizer.normalize(text,Normalizer.Form.NFC)
    val clean=if nuiFallbackNormalization then nfc else spaces.matcher(nfc).replaceAll(" ")
    // HF Lowercase expands each code point independently (no contextual final sigma).
    if nuiFallbackNormalization then normalizedBuilder.append(clean)
    else clean.codePoints().forEach(cp => normalizedBuilder.append(new String(Character.toChars(cp)).toLowerCase(Locale.ROOT)))
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
      val prepared=if !nuiFallbackNormalization then segment else
        val builder=new java.lang.StringBuilder()
        segment.codePoints().forEach { cp =>
          val kind=Character.getType(cp)
          if cp==0 || cp==0xfffd || ((kind==Character.CONTROL || kind==Character.FORMAT) && cp!=9 && cp!=10 && cp!=13) then ()
          else if cp==9 || cp==10 || cp==13 || kind==Character.SPACE_SEPARATOR then builder.append(' ')
          else if isChinese(cp) then { builder.append(' ');builder.appendCodePoint(cp);builder.append(' ') }
          else builder.appendCodePoint(cp)
        }
        builder.toString.toLowerCase(Locale.ROOT)
      val matcher=words.matcher(prepared)
      while matcher.find() do piece(matcher.group())
    val matcher=addedTokenPattern.matcher(normalized)
    var start=0
    while matcher.find() do
      ordinary(normalized.substring(start,matcher.start()))
      val token=if nuiFallbackNormalization then matcher.group().toLowerCase(Locale.ROOT) else matcher.group()
      vectorGroups.get(token) match
        case Some(rows) => rows.foreach(name => output+=addedTokenIds(name))
        case None => output+=vocab.get(token).map(_.toLong).orElse(addedTokenIds.get(token))
          .getOrElse(throw IllegalStateException(s"missing CLIP special token: $token"))
      start=matcher.end()
    ordinary(normalized.substring(start))
    if addSpecialTokens then output+=eosId
    output.toArray
  def batch(texts: Seq[String],truncate: Boolean=false,maxElements: Long=1048576L): ClipTokenBatch =
    val count=texts.size.toLong*maxLength
    require(maxElements>=0 && count<=maxElements && count<=Int.MaxValue,"CLIP batch exceeds allocation limit")
    val ids=Array.fill(count.toInt)(padId)
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
  private def isChinese(cp: Int): Boolean =
    (cp>=0x4e00 && cp<=0x9fff) || (cp>=0x3400 && cp<=0x4dbf) ||
    (cp>=0x20000 && cp<=0x2a6df) || (cp>=0x2a700 && cp<=0x2b73f) ||
    (cp>=0x2b740 && cp<=0x2b81f) || (cp>=0x2b820 && cp<=0x2ceaf) ||
    (cp>=0xf900 && cp<=0xfaff) || (cp>=0x2f800 && cp<=0x2fa1f)
  private val alphabet: Vector[String]=
    var next=256
    (0 to 255).map { b =>
      if (b>=33 && b<=126) || (b>=161 && b<=172) || b>=174 then b.toChar.toString
      else { val s=next.toChar.toString;next+=1;s }
    }.toVector
  private lazy val inverseAlphabet: Map[Char,Byte]=alphabet.zipWithIndex.map((symbol,index)=>symbol.charAt(0)->index.toByte).toMap
  private def read(path: Path): String=Using.resource(Files.newInputStream(path)) { stream =>
    val bytes=stream.readNBytes(TokenizerJson.maxBytes+1)
    require(bytes.length<=TokenizerJson.maxBytes,"CLIP tokenizer file exceeds32MiB")
    UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString
  }
  def load(vocabPath: Path,mergesPath: Path,maxLength: Int=77,padToken: String=eos): ClipFastTokenizer =
    fromText(read(vocabPath),read(mergesPath),maxLength,padToken)
  /** Directory profile including explicit CLIP special-token padding configuration. */
  def loadDirectory(directory: Path): ClipFastTokenizer =
    val path=directory.resolve("tokenizer_config.json")
    val config=if Files.exists(path) then TokenizerJson.read(path).obj else Map.empty[String,TokenizerJson.Value]
    def token(key: String,default: String): String=config.get(key) match
      case None | Some(TokenizerJson.Value.Null) => default
      case Some(TokenizerJson.Value.Str(value)) => value
      case Some(value) => value.obj("content").str
    require(token("bos_token",bos)==bos && token("eos_token",eos)==eos && token("unk_token",eos)==eos,
      "unsupported CLIP special token configuration")
    val length=config.get("model_max_length").fold(77)(_.int)
    load(directory.resolve("vocab.json"),directory.resolve("merges.txt"),length,token("pad_token",eos))
  def fromText(vocabJson: String,mergesText: String,maxLength: Int=77,padToken: String=eos): ClipFastTokenizer =
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
    require(vocab.contains(padToken),"missing CLIP pad token")
    new ClipFastTokenizer(vocab,pairs.zipWithIndex.toMap,maxLength,vocab(padToken).toLong)
