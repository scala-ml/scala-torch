package scalatorch.models

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path,NoSuchFileException}
import java.nio.file.attribute.BasicFileAttributes
import scala.collection.mutable

/** Reads fresh snapshots so custom mapping edits become visible on the next lookup.
  * No files are created or edited by lookup. A concurrent file replacement fails
  * the read instead of publishing a mixture of old and new contents.
  */
final class KeywordMappings(defaultPath: Path,userPath: Path,allowMissingDefault: Boolean=false):
  def lookup(hash: String,filename: String,check: () => Unit=() => ()): Option[KeywordMappings.Entry]=
    val defaults=KeywordMappings.read(defaultPath,0,allowMissingDefault,check)
    val custom=KeywordMappings.read(userPath,1,true,check)
    KeywordMappings.find(defaults++custom,hash,filename,check)

object KeywordMappings:
  final case class Entry(hash: String,keywords: String,filename: String,source: Int)
  private val maxBytes=64*1024*1024
  private def whitespace(c: Char): Boolean=Character.isWhitespace(c) || Character.isSpaceChar(c) || c=='\u0085'
  private def stripped(text: String): String=text.dropWhile(whitespace).reverse.dropWhile(whitespace).reverse
  private def lower(text: String): String=text.toLowerCase(java.util.Locale.ROOT)

  /** CSV dialect used by NUI: comma separator, quoted multiline fields and skipinitialspace. */
  def parse(text: String,source: Int,check: () => Unit=() => ()): Vector[Entry]=
    require(text.length<=maxBytes,"keyword mapping exceeds 64MiB")
    val entries=Vector.newBuilder[Entry]
    val row=mutable.ArrayBuffer.empty[String]
    val field=new StringBuilder
    var quoted=false
    var atStart=true
    var touched=false
    var entryCount=0
    def endField(): Unit=
      require(row.size<256,"keyword CSV row exceeds 256 fields")
      row+=field.toString
      field.clear()
      atStart=true
    def endRow(): Unit=
      endField()
      if row.size>=2 then
        val hash=stripped(row(0))
        if hash.nonEmpty && !hash.startsWith("#") then
          val name=if row.size>=3 then stripped(row(2)) else "default"
          require(name.codePointCount(0,name.length)<=4096,"keyword filename exceeds 4096 codepoints")
          require(entryCount<1000000,"keyword mapping exceeds one million entries")
          entries+=Entry(lower(hash),stripped(row(1)),name,source)
          entryCount+=1
      row.clear()
      touched=false
    var i=0
    while i<text.length do
      if (i&65535)==0 then check()
      val c=text.charAt(i)
      if quoted then
        if c=='"' then
          if i+1<text.length && text.charAt(i+1)=='"' then
            field.append('"');i+=1
          else quoted=false
        else field.append(c)
      else if c==',' then
        endField();touched=true
      else if c=='\r' || c=='\n' then
        endRow()
        if c=='\r' && i+1<text.length && text.charAt(i+1)=='\n' then i+=1
      else if atStart && c==' ' then touched=true
      else if atStart && c=='"' then
        quoted=true;atStart=false;touched=true
      else
        field.append(c);atStart=false;touched=true
      require(field.length<=131072,"keyword CSV field exceeds 128KiB")
      i+=1
    if touched || row.nonEmpty || field.nonEmpty then endRow()
    check()
    entries.result()

  private def attributes(path: Path): BasicFileAttributes=Files.readAttributes(path,classOf[BasicFileAttributes])
  private def unchanged(path: Path,before: BasicFileAttributes): Unit=
    val after=attributes(path)
    require(before.size()==after.size() && before.lastModifiedTime()==after.lastModifiedTime() &&
      before.fileKey()==after.fileKey(),s"keyword source changed while reading: $path")
  private def read(path: Path,source: Int,missingAllowed: Boolean,check: () => Unit): Vector[Entry]=
    val before=try Some(attributes(path)) catch
      case _: NoSuchFileException if missingAllowed => None
    before match
      case None => Vector.empty
      case Some(stat) =>
        require(stat.isRegularFile && stat.size()<=maxBytes,"keyword source must be a regular file up to 64MiB")
        val bytes=scala.util.Using.resource(Files.newInputStream(path)) { input =>
          val out=new java.io.ByteArrayOutputStream
          val buffer=new Array[Byte](65536)
          var count=input.read(buffer)
          while count != -1 do
            check()
            require(out.size().toLong+count<=maxBytes,"keyword source exceeds 64MiB")
            out.write(buffer,0,count)
            count=input.read(buffer)
          out.toByteArray
        }
        unchanged(path,stat)
        parse(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString,source,check)

  def legacyHash(path: Path,check: () => Unit=() => ()): String=
    val before=attributes(path)
    require(before.isRegularFile,"legacy keyword hash requires a checkpoint file")
    val digest=java.security.MessageDigest.getInstance("SHA-256")
    scala.util.Using.resource(Files.newByteChannel(path)) { channel =>
      channel.position(0x100000)
      val buffer=ByteBuffer.allocate(0x10000)
      var done=false
      while buffer.hasRemaining && !done do
        check()
        if channel.read(buffer)<0 then done=true
      buffer.flip()
      digest.update(buffer)
    }
    unchanged(path,before)
    java.util.HexFormat.of().formatHex(digest.digest()).take(8)

  def find(rows: Vector[Entry],hash: String,filename: String,check: () => Unit=() => ()): Option[Entry]=
    val wanted=lower(hash)
    var result=Option.empty[Entry]
    var best = -1.0
    for row <- rows if row.hash==wanted do
      check()
      val score=similarity(row.filename,filename,check)
      if score>=best then
        best=score;result=Some(row)
    result

  /** SequenceMatcher(None, a, b).ratio(): ordered longest blocks with autojunk.
    * Unicode codepoints, rather than UTF-16 units, match Python string semantics.
    */
  private[models] def similarity(left: String,right: String,check: () => Unit=() => ()): Double=
    val a=left.codePoints().toArray
    val b=right.codePoints().toArray
    require(a.length<=4096 && b.length<=4096,"keyword filename exceeds 4096 codepoints")
    if a.isEmpty && b.isEmpty then return 1.0
    val positions=b.indices.groupBy(b(_)).filter((_,indices) => b.length<200 || indices.size<=b.length/100+1)
    val pending=mutable.ArrayDeque((0,a.length,0,b.length))
    var matched=0
    while pending.nonEmpty do
      val (alo,ahi,blo,bhi)=pending.removeLast()
      var startA=alo
      var startB=blo
      var size=0
      var previous=Map.empty[Int,Int]
      for i <- alo until ahi do
        check()
        val current=mutable.Map.empty[Int,Int]
        for j <- positions.getOrElse(a(i),Vector.empty) if j>=blo && j<bhi do
          val length=previous.getOrElse(j-1,0)+1
          current(j)=length
          if length>size then
            startA=i-length+1;startB=j-length+1;size=length
        previous=current.toMap
      while startA>alo && startB>blo && a(startA-1)==b(startB-1) do
        startA-=1;startB-=1;size+=1
      while startA+size<ahi && startB+size<bhi && a(startA+size)==b(startB+size) do size+=1
      if size>0 then
        matched+=size
        if alo<startA && blo<startB then pending.append((alo,startA,blo,startB))
        if startA+size<ahi && startB+size<bhi then pending.append((startA+size,ahi,startB+size,bhi))
    2.0*matched/(a.length+b.length)
