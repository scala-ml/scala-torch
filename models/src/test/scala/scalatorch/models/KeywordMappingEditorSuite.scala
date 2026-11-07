package scalatorch.models

import java.nio.file.Files

class KeywordMappingEditorSuite extends munit.FunSuite:
  test("edit preserves unrelated bytes removes duplicate keys and quotes new values") {
    val original="# keep\r\nabc,old,target\r\nabc,duplicate,target\nxyz,\"multi\nline\",other\n"
    val (updated,found)=KeywordMappingEditor.rewrite(original,"ABC","target",Some("red, blue|green"))
    assert(found)
    assertEquals(updated,"# keep\r\nxyz,\"multi\nline\",other\nabc,\"red, blue|green\",target\n")
    val (deleted,removed)=KeywordMappingEditor.rewrite(updated,"abc","target",None)
    assert(removed)
    assertEquals(deleted,"# keep\r\nxyz,\"multi\nline\",other\n")
    assertEquals(KeywordMappingEditor.rewrite(deleted,"abc","target",None),(deleted,false))
    intercept[IllegalArgumentException](KeywordMappingEditor.rewrite("abc,\"unclosed,target","abc","target",None))
    intercept[IllegalArgumentException](KeywordMappingEditor.rewrite("abc,\"closed\" junk,target","abc","target",None))
  }
  test("atomic edit backs up original bytes and rejected publication preserves original") {
    val directory=Files.createTempDirectory("keyword-editor")
    val path=directory.resolve("custom-mappings.txt")
    val original="# keep\r\nabc,old,target\r\n"
    try
      Files.writeString(path,original)
      intercept[java.util.concurrent.CancellationException](KeywordMappingEditor.edit(path,"abc","target",Some("new"),
        publish=() => throw new java.util.concurrent.CancellationException))
      assertEquals(Files.readString(path),original)
      assert(KeywordMappingEditor.edit(path,"abc","target",Some("new")))
      assertEquals(Files.readString(path),"# keep\r\nabc,new,target\n")
      val files=Files.list(directory)
      val backups=try
        import scala.jdk.CollectionConverters.*
        files.iterator().asScala.filter(_.getFileName.toString.startsWith("custom-mappings.txt.backup-")).toVector
      finally files.close()
      assertEquals(backups.size,1)
      assertEquals(Files.readString(backups.head),original)
    finally
      val files=Files.list(directory)
      try files.forEach(p => {Files.delete(p);()}) finally files.close()
      Files.delete(directory)
  }
  test("lock contention is cancellable and malformed CSV remains untouched") {
    val directory=Files.createTempDirectory("keyword-lock")
    val path=directory.resolve("custom.csv")
    val lockPath=directory.resolve("custom.csv.lock")
    try
      Files.writeString(path,"abc,\"unterminated")
      intercept[IllegalArgumentException](KeywordMappingEditor.edit(path,"abc","target",Some("new")))
      assertEquals(Files.readString(path),"abc,\"unterminated")
      val channel=java.nio.channels.FileChannel.open(lockPath,java.nio.file.StandardOpenOption.WRITE)
      try
        val lock=channel.lock(0,1,false)
        try
          var calls=0
          intercept[java.util.concurrent.CancellationException](KeywordMappingEditor.edit(path,"abc","target",Some("new"),check=() => {
            calls+=1
            if calls>8 then throw new java.util.concurrent.CancellationException
          }))
        finally lock.release()
      finally channel.close()
      assertEquals(Files.readString(path),"abc,\"unterminated")
    finally
      Files.deleteIfExists(lockPath);Files.deleteIfExists(path);Files.delete(directory)
  }
