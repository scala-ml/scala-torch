package scalatorch.models

import scalatorch.*
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.{CharacterCodingException,StandardCharsets}
import java.nio.file.{Files,Path}
import java.security.MessageDigest
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

final case class TrainingBatch(input: Tensor[Long],targets: Tensor[Long],targetCount: Int)

/** Deterministic packed corpus. Each transition appears once per epoch; padding targets are ignored. */
final class TextDataset private (tokens: Array[Long],val vocabularySize: Int,val eosToken: Long,
    val sequenceLength: Int,val batchSize: Int):
  val tokenCount: Int=tokens.length
  private val blocks=(tokens.length-1L+sequenceLength-1)/sequenceLength
  val batchesPerEpoch: Long=(blocks+batchSize-1)/batchSize
  val fingerprint: String =
    val hash=MessageDigest.getInstance("SHA-256")
    val buffer=ByteBuffer.allocate(8192).order(ByteOrder.BIG_ENDIAN)
    def add(n: Long): Unit =
      if buffer.remaining()<8 then
        buffer.flip();hash.update(buffer);buffer.clear()
      buffer.putLong(n)
    add(vocabularySize);add(eosToken);add(tokens.length)
    tokens.foreach(add)
    buffer.flip();hash.update(buffer)
    hash.digest().map(b => f"${b & 255}%02x").mkString
  def batch(step: Long): TrainingBatch =
    require(step>=0,"negative training step")
    val size=batchSize*sequenceLength
    val input=Array.fill(size)(eosToken)
    val targets=Array.fill(size)(-100L)
    val firstBlock=(step%batchesPerEpoch)*batchSize
    var count=0
    var row=0
    while row<batchSize do
      val start=(firstBlock+row)*sequenceLength
      val available=math.max(0L,math.min(sequenceLength.toLong,tokens.length-1L-start)).toInt
      var col=0
      while col<available do
        input(row*sequenceLength+col)=tokens((start+col).toInt)
        targets(row*sequenceLength+col)=tokens((start+col+1).toInt)
        col+=1
      count+=available;row+=1
    val shape=Shape.from(Seq(batchSize.toLong,sequenceLength.toLong))
    TrainingBatch(Tensor.fromLongs(input,shape),Tensor.fromLongs(targets,shape),count)

object TextDataset:
  def fromTokens(values: Array[Long],vocabularySize: Int,eosToken: Long,sequenceLength: Int,batchSize: Int,
      maxTokens: Int = 1048576): TextDataset =
    require(vocabularySize>0 && eosToken>=0 && eosToken<vocabularySize,"invalid dataset vocabulary/EOS")
    require(maxTokens>=2 && values.length>=2 && values.length<=maxTokens,"dataset token limit or insufficient tokens")
    require(sequenceLength>0 && batchSize>0 && sequenceLength.toLong*batchSize<=maxTokens,"invalid dataset batch dimensions")
    require(values.forall(id => id>=0 && id<vocabularySize),"dataset token outside vocabulary")
    new TextDataset(values.clone(),vocabularySize,eosToken,sequenceLength,batchSize)
  /** Nonempty UTF-8 lines are documents. Explicit trailing EOS tokens are not duplicated. */
  def read(path: Path,tokenizer: ByteBpeTokenizer,eosToken: Long,sequenceLength: Int,batchSize: Int,
      maxBytes: Int = 8388608,maxTokens: Int = 1048576): TextDataset =
    require(maxBytes>0 && maxBytes<Int.MaxValue && maxTokens>=2,"invalid dataset read limits")
    require(eosToken>=0 && eosToken<tokenizer.vocabSize,"invalid dataset EOS")
    val bytes=Using.resource(Files.newInputStream(path))(_.readNBytes(maxBytes+1))
    require(bytes.length<=maxBytes,"dataset byte limit exceeded")
    val text=try StandardCharsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString
      catch case error: CharacterCodingException => throw IllegalArgumentException("dataset is not valid UTF-8",error)
    val tokens=ArrayBuffer.empty[Long]
    text.linesIterator.filter(_.nonEmpty).foreach { line =>
      val encoded=tokenizer.encode(line)
      require(encoded.nonEmpty,"document has no supported tokens")
      val append=encoded.last!=eosToken
      require(tokens.size.toLong+encoded.length+(if append then 1 else 0)<=maxTokens,"dataset token limit exceeded")
      tokens++=encoded
      if append then tokens+=eosToken
    }
    fromTokens(tokens.toArray,tokenizer.vocabSize,eosToken,sequenceLength,batchSize,maxTokens)
