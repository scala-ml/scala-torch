package scalatorch.models

import java.nio.ByteBuffer
import java.nio.channels.{FileChannel,FileLock,OverlappingFileLockException}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path,NoSuchFileException,StandardOpenOption,StandardCopyOption}

/** Lossless CSV edits under the same one-byte Windows lock used by NUI. */
object KeywordMappingEditor:
  private val limit=64*1024*1024
  private def strip(text: String): String=
    def space(c: Char)=Character.isWhitespace(c) || Character.isSpaceChar(c) || c=='\u0085'
    text.dropWhile(space).reverse.dropWhile(space).reverse
  def rewrite(original: String,hash: String,filename: String,keywords: Option[String],check: () => Unit=() => ()): (String,Boolean)=
    val normalized=strip(hash).toLowerCase(java.util.Locale.ROOT)
    require(normalized.nonEmpty && strip(filename).nonEmpty,"model hash and filename are required")
    require(keywords.forall(k => strip(k).nonEmpty),"keywords are required")
    require(original.length<=limit,"mapping exceeds 64MiB")
    val retained=new StringBuilder
    var found=false
    var start=0
    var columns=1
    var quoted=false
    var afterQuote=false
    var atStart=true
    def record(end: Int): Unit=
      val raw=original.substring(start,end)
      val entry=KeywordMappings.parse(raw,1,check).headOption
      val matches=columns>=3 && entry.exists(e => e.hash==normalized && e.filename==filename)
      if matches then found=true else retained.append(raw)
      start=end;columns=1;atStart=true;afterQuote=false
    var i=0
    while i<original.length do
      if (i&65535)==0 then check()
      val c=original.charAt(i)
      if quoted then
        if c=='"' then
          if i+1<original.length && original.charAt(i+1)=='"' then i+=1
          else
            quoted=false;afterQuote=true
      else if c==',' then
        columns+=1;atStart=true;afterQuote=false
        require(columns<=256,"mapping CSV exceeds 256 columns")
      else if c=='\r' || c=='\n' then
        if c=='\r' && i+1<original.length && original.charAt(i+1)=='\n' then i+=1
        record(i+1)
      else
        require(!afterQuote,"cannot edit malformed mapping CSV after closing quote")
        if atStart && c=='"' then
          quoted=true;atStart=false
        else if !(atStart && c==' ') then atStart=false
      i+=1
    require(!quoted,"cannot edit unterminated mapping CSV quote")
    if start<original.length then record(original.length)
    def csv(text: String): String=
      if text.exists(c => c==',' || c=='"' || c=='\r' || c=='\n') then "\""+text.replace("\"","\"\"")+"\"" else text
    keywords.foreach { value =>
      if retained.nonEmpty && retained.last!='\n' && retained.last!='\r' then retained.append('\n')
      retained.append(Vector(normalized,strip(value),filename).map(csv).mkString(",")).append('\n')
    }
    val result=retained.toString
    require(result.length<=limit,"edited mapping exceeds 64MiB")
    // Validate new values against the reader's field and filename budgets too.
    KeywordMappings.parse(result,1,check)
    (result,found)

  private def bytes(path: Path): Option[Array[Byte]]=
    try Some(scala.util.Using.resource(Files.newInputStream(path)) { input =>
      val data=input.readNBytes(limit+1)
      require(data.length<=limit,"mapping exceeds 64MiB")
      data
    }) catch case _: NoSuchFileException => None

  def edit(file: Path,hash: String,filename: String,keywords: Option[String],
      check: () => Unit=() => (),publish: () => Unit=() => ()): Boolean=
    check()
    // Validate before creating even the lock directory.
    rewrite("",hash,filename,keywords,check)
    val path=file.toAbsolutePath.normalize()
    Files.createDirectories(path.getParent)
    val lockPath=path.resolveSibling(path.getFileName.toString+".lock")
    scala.util.Using.resource(FileChannel.open(lockPath,StandardOpenOption.CREATE,StandardOpenOption.READ,StandardOpenOption.WRITE)) { channel =>
      val deadline=System.nanoTime()+30L*1000000000L
      var lock: FileLock=null
      while lock==null do
        check()
        try lock=channel.tryLock(0,1,false) catch case _: OverlappingFileLockException => ()
        if lock==null then
          require(System.nanoTime()<deadline,"timed out waiting for keyword mapping lock")
          Thread.sleep(10)
      try
        val original=bytes(path)
        val text=original.fold("")(data => UTF_8.newDecoder().decode(ByteBuffer.wrap(data)).toString)
        val (updated,found)=rewrite(text,hash,filename,keywords,check)
        if keywords.isEmpty && !found then
          publish()
          false
        else
          val output=updated.getBytes(UTF_8)
          require(output.length<=limit,"edited mapping exceeds 64MiB")
          val temporary=Files.createTempFile(path.getParent,".model-keyword-",".tmp")
          var backup=Option.empty[Path]
          var committed=false
          try
            def write(target: Path,data: Array[Byte]): Unit=
              scala.util.Using.resource(FileChannel.open(target,StandardOpenOption.WRITE,StandardOpenOption.TRUNCATE_EXISTING)) { out =>
                val buffer=ByteBuffer.wrap(data)
                while buffer.hasRemaining do
                  check();out.write(buffer)
                out.force(true)
              }
            write(temporary,output)
            original.foreach { data =>
              val saved=Files.createTempFile(path.getParent,path.getFileName.toString+".backup-","")
              backup=Some(saved)
              write(saved,data)
            }
            val current=bytes(path)
            require((original,current) match
              case (None,None) => true
              case (Some(a),Some(b)) => java.util.Arrays.equals(a,b)
              case _ => false
            ,"mapping changed during edit")
            publish()
            Files.move(temporary,path,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
            committed=true
            found
          finally
            Files.deleteIfExists(temporary)
            if !committed then backup.foreach(Files.deleteIfExists(_))
      finally lock.release()
    }
