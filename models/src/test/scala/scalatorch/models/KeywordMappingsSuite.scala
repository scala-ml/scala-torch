package scalatorch.models

import java.nio.file.Files

class KeywordMappingsSuite extends munit.FunSuite:
  import KeywordMappings.*
  test("CSV and SequenceMatcher match static Python fixtures including Unicode and autojunk") {
    import scalatorch.models.worker.NativeJson
    import NativeJson.*
    def fixtures(name: String): Vector[Map[String,NativeJson]]=
      val input=scala.io.Source.fromInputStream(this.getClass.getResourceAsStream(s"/model-keyword/$name.jsonl"),"UTF-8")
      try input.getLines().map(line => NativeJson.parse(line).asInstanceOf[Obj].fields).toVector
      finally input.close()
    val ratios=fixtures("similarity")
    assertEquals(ratios.size,160)
    for row <- ratios do
      assertEqualsDouble(similarity(row("left").asInstanceOf[Str].value,row("right").asInstanceOf[Str].value),
        row("ratio").asInstanceOf[Number].value.toDouble,1e-15)
    val csv=fixtures("csv")
    assertEquals(csv.size,10)
    for row <- csv do
      val expected=row("entries").asInstanceOf[Arr].values.map { entry =>
        val fields=entry.asInstanceOf[Obj].fields
        Entry(fields("model_hash").asInstanceOf[Str].value,fields("keywords").asInstanceOf[Str].value,
          fields("filename").asInstanceOf[Str].value,fields("source").asInstanceOf[Number].value.toInt)
      }
      assertEquals(KeywordMappings.parse(row("text").asInstanceOf[Str].value,1),expected)
  }
  test("CSV retains quoted commas newlines escaped quotes and empty keywords") {
    val parsed=parse("# comment\r\n AB12, \"red, blue|green\",similar.safetensors\r\n"+
      "ab12,\"line1\nline2 \"\"quoted\"\"\",name\nempty,,\nshort,words\nignored\n",0)
    assertEquals(parsed,Vector(Entry("ab12","red, blue|green","similar.safetensors",0),
      Entry("ab12","line1\nline2 \"quoted\"","name",0),Entry("empty","","",0),Entry("short","words","default",0)))
  }
  test("matching uses filename similarity then later row on a tie") {
    val rows=Vector(Entry("ab12","base","similar.safetensors",0),Entry("ab12","custom","similar.safetensors",1),
      Entry("ab12","unrelated","other.ckpt",1))
    assertEquals(find(rows,"AB12","similar.safetensors").map(_.keywords),Some("custom"))
    assertEquals(find(rows,"absent","similar.safetensors"),None)
    assertEquals(find(rows.take(1)++rows.drop(2),"ab12","similar.safetensors").map(_.keywords),Some("base"))
  }
  test("index observes custom changes and deletion without stale cache") {
    val directory=Files.createTempDirectory("keyword-mappings")
    val defaults=directory.resolve("defaults.csv")
    val custom=directory.resolve("custom.csv")
    try
      Files.writeString(defaults,"abc,base,model.safetensors\n")
      val index=new KeywordMappings(defaults,custom)
      assertEquals(index.lookup("abc","model.safetensors").map(_.keywords),Some("base"))
      Files.writeString(custom,"abc,user,model.safetensors\n")
      assertEquals(index.lookup("abc","model.safetensors").map(_.source),Some(1))
      Files.writeString(custom,"abc,edit,model.safetensors\n")
      assertEquals(index.lookup("abc","model.safetensors").map(_.keywords),Some("edit"))
      Files.delete(custom)
      assertEquals(index.lookup("abc","model.safetensors").map(_.keywords),Some("base"))
      Files.delete(defaults)
      intercept[java.nio.file.NoSuchFileException](index.lookup("abc","model.safetensors"))
      assertEquals(new KeywordMappings(defaults,custom,allowMissingDefault=true).lookup("abc","model.safetensors"),None)
    finally
      Files.deleteIfExists(custom);Files.deleteIfExists(defaults);Files.delete(directory)
  }
  test("legacy checkpoint hash reads only 64KiB at one MiB") {
    val file=Files.createTempFile("keyword-legacy",".safetensors")
    try
      val bytes=Array.fill[Byte](0x100000+0x10000+1)(0)
      for i <- 0 until 0x10000 do bytes(0x100000+i)=i.toByte
      Files.write(file,bytes)
      val digest=java.security.MessageDigest.getInstance("SHA-256").digest(bytes.slice(0x100000,0x110000))
      assertEquals(legacyHash(file),java.util.HexFormat.of().formatHex(digest).take(8))
      Files.write(file,Array[Byte](1,2,3))
      assertEquals(legacyHash(file),"e3b0c442")
    finally Files.delete(file)
  }
  test("mapping reads reject malformed UTF8 and concurrent edits and support cancellation") {
    val file=Files.createTempFile("keyword-snapshot",".csv")
    val missing=file.resolveSibling(file.getFileName.toString+".absent")
    val index=new KeywordMappings(file,missing)
    try
      Files.write(file,Array[Byte](0xc3.toByte,0x28))
      intercept[java.nio.charset.CharacterCodingException](index.lookup("a","name"))
      Files.writeString(file,"a,words,name\n")
      intercept[java.util.concurrent.CancellationException](index.lookup("a","name",() => throw new java.util.concurrent.CancellationException))
      var changed=false
      intercept[IllegalArgumentException](index.lookup("a","name",() => {
        if !changed then
          changed=true
          Files.writeString(file,"a,changed words,name\n")
      }))
      assertEquals(index.lookup("a","name").map(_.keywords),Some("changed words"))
      assert(!Files.exists(missing))
    finally Files.delete(file)
  }
  test("CSV resource bounds fail before accumulating excessive fields") {
    intercept[IllegalArgumentException](parse("a,"+("x"*131073),0))
    intercept[IllegalArgumentException](parse("a,"+("x,"*256),0))
  }
