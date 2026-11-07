package scalatorch.models

import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer

final case class PromptSpan(text: String,weight: Double)
final case class PromptChunk(tokens: Vector[Long],weights: Vector[Double])

/** NUI attention syntax and fixed-length CLIP chunks; no tokenizer or Python dependency. */
object DiffusionPrompt:
  private val attention=Pattern.compile("""\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*(x?[+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:""",Pattern.UNICODE_CHARACTER_CLASS)
  private val breaks=Pattern.compile("""\s*\bBREAK\b\s*""",Pattern.UNICODE_CHARACTER_CLASS)
  private val maxText=1048576
  private val maxSpans=65536

  def parse(text: String): Vector[PromptSpan] =
    require(text.length<=maxText && TokenizerJson.validUnicode(text),"invalid or oversized prompt")
    val spans=ArrayBuffer.empty[PromptSpan]
    val rounds=ArrayBuffer.empty[Int]
    val squares=ArrayBuffer.empty[Int]
    var characters=0L
    def append(s: String,w: Double): Unit =
      characters+=s.length
      require(spans.size<maxSpans && characters<=maxText,"prompt expansion exceeds limit")
      spans+=PromptSpan(s,w)
    def multiply(start: Int,factor: Double): Unit =
      require(factor.isFinite,"nonfinite prompt weight")
      for i <- start until spans.size do
        val updated=spans(i).weight*factor
        require(updated.isFinite,"prompt weight overflow")
        spans(i)=spans(i).copy(weight=updated)
    def pop(stack: ArrayBuffer[Int]): Int=stack.remove(stack.size-1)
    val matcher=attention.matcher(text)
    while matcher.find() do
      val token=matcher.group()
      val explicit=matcher.group(1)
      if token.startsWith("\\") then append(token.substring(1),1)
      else if token=="(" || token=="[" then
        require(rounds.size+squares.size<4096,"prompt nesting exceeds limit")
        (if token=="(" then rounds else squares)+=spans.size
      else if explicit!=null && rounds.nonEmpty then
        if explicit.startsWith("x") then
          explicit.substring(1).toDoubleOption match
            case None => append(token,1)
            case Some(value) =>
              require(value.isFinite && math.abs(value)<=maxSpans,"prompt repeat exceeds limit")
              val start=pop(rounds)
              val segment=spans.drop(start).toVector
              val count=math.max(1,math.abs(value).toInt)
              val tail=math.abs(value)%1+(if math.abs(value)>=1 then 1 else 0)
              require(start.toLong+segment.size.toLong*count<=maxSpans,"prompt repeat exceeds limit")
              val length=segment.map(_.text.length.toLong).sum
              require(characters+length*(count-1)<=maxText,"prompt expansion exceeds limit")
              spans.dropRightInPlace(spans.size-start)
              characters-=length
              for repetition <- 0 until count; span <- segment do
                val factor=(if repetition==count-1 && tail>1e-4 then tail else 1)*(if value<0 then -1 else 1)
                append(span.text,span.weight)
                multiply(spans.size-1,factor)
        else
          val start=pop(rounds)
          explicit.toDoubleOption match
            case Some(weight) => multiply(start,weight)
            case None => append(token,1)
      else if token==")" && rounds.nonEmpty then multiply(pop(rounds),1.1)
      else if token=="]" && squares.nonEmpty then multiply(pop(squares),1/1.1)
      else
        breaks.split(token,-1).zipWithIndex.foreach { (part,index) =>
          if index>0 then append("BREAK",-1)
          append(part,1)
        }
    rounds.foreach(multiply(_,1.1))
    squares.foreach(multiply(_,1/1.1))
    if spans.isEmpty then append("",1)
    // Assemble equal-weight runs once, avoiding quadratic concatenation for long prompts.
    val merged=Vector.newBuilder[PromptSpan]
    var start=0
    while start<spans.size do
      val builder=new java.lang.StringBuilder(spans(start).text)
      var end=start+1
      while end<spans.size && spans(end).weight==spans(start).weight do
        builder.append(spans(end).text)
        end+=1
      merged+=PromptSpan(builder.toString,spans(start).weight)
      start=end
    merged.result()

  def chunks(spans: Seq[PromptSpan],tokenize: String => Array[Long],bos: Long,eos: Long,pad: Long,
      comma: Option[Long]=None,contentLength: Int=75,commaBacktrack: Int=20): Vector[PromptChunk] =
    require(contentLength>0 && contentLength<=16382 && commaBacktrack>=0,"invalid prompt chunk dimensions")
    require(spans.size<=maxSpans && spans.map(_.text.length.toLong).sum<=maxText,"prompt exceeds limit")
    val result=ArrayBuffer.empty[PromptChunk]
    var tokens=ArrayBuffer.empty[Long]
    var weights=ArrayBuffer.empty[Double]
    var lastComma= -1
    def finish(): Unit =
      require(result.size<256,"prompt exceeds 256 chunks")
      val ids=Vector(bos)++tokens.toVector++Vector.fill(contentLength-tokens.size+1)(eos)
      val firstEos=ids.indexOf(eos,1)
      val padded=if pad!=eos && firstEos>=0 then ids.take(firstEos+1)++Vector.fill(ids.size-firstEos-1)(pad) else ids
      result+=PromptChunk(padded,Vector(1.0)++weights.toVector++Vector.fill(contentLength-weights.size+1)(1.0))
      tokens=ArrayBuffer.empty[Long];weights=ArrayBuffer.empty[Double];lastComma= -1
    spans.foreach { span =>
      require(span.weight.isFinite,"nonfinite prompt weight")
      if span.text=="BREAK" && span.weight== -1 then finish()
      else
        val encoded=tokenize(span.text)
        require(encoded.length<=maxText,"prompt tokenization exceeds limit")
        encoded.foreach { token =>
          if comma.contains(token) then lastComma=tokens.size
          else if commaBacktrack!=0 && tokens.size==contentLength && lastComma>=0 && tokens.size-lastComma<=commaBacktrack then
            val relocated=tokens.drop(lastComma+1)
            val relocatedWeights=weights.drop(lastComma+1)
            tokens.dropRightInPlace(tokens.size-lastComma-1);weights.dropRightInPlace(weights.size-lastComma-1)
            finish()
            tokens=relocated;weights=relocatedWeights
          if tokens.size==contentLength then finish()
          tokens+=token;weights+=span.weight
        }
    }
    if tokens.nonEmpty || result.isEmpty then finish()
    result.toVector
