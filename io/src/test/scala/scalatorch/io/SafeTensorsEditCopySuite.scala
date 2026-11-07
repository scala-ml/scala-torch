package scalatorch.io

import java.nio.file.{Files,Path}
import java.security.MessageDigest
import java.util.concurrent.CancellationException

class SafeTensorsEditCopySuite extends munit.FunSuite:
  private def fixture(name: String,hash: String): Array[Byte] =
    val stream=getClass.getResourceAsStream(s"/safetensors/$name")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)),hash)
    bytes
  private val sourceHash="ab884d6ddd496ef99f51eca509e7a74cb555e3df8e513e803470d7f7c8597c2a"
  private val expectedHash="cd23ca8155dfc2edff7ab2c183568b2f5add3331915b5a83502f7b52419240e6"
  private def baseline=Map("half"->Vector(Vector(1.0,-2.0),Vector(.25,10.0)),
    "bfloat"->Vector(Vector(.5,-4.0),Vector(1.25,2.0)),"single"->Vector(Vector(3.0,-5.0)),
    "double"->Vector(Vector(1.25,-2.5)))
  private def edited=baseline.updated("half",Vector(Vector(1.0,-2.0),Vector(.25,9.5)))
    .updated("bfloat",Vector(Vector(.5,-3.5),Vector(1.25,2.0)))
    .updated("single",Vector(Vector(3.25,-5.0))).updated("double",Vector(Vector(1.25,-2.25)))
  test("editor copy preserves all unedited bytes exactly as NUI") {
    val dir=Files.createTempDirectory("embedding-edit-copy")
    try
      val source=dir.resolve("source.safetensors");val target=dir.resolve("stage.safetensors")
      val original=fixture("edit-source.safetensors",sourceHash);Files.write(source,original)
      SafeTensorsEditCopy.build(source,target,edited,sourceHash,Some(baseline))
      assertEquals(Files.readAllBytes(target).toVector,fixture("edit-expected.safetensors",expectedHash).toVector)
      assertEquals(Files.readAllBytes(source).toVector,original.toVector)
      val archive=SafeTensors.open(target)
      try
        assertEquals(archive.metadata,Map("format"->"pt","note"->"unchanged metadata"))
        assertEquals(archive.readLong("aux").toLongArray.toVector,Vector(4L,-7L,19L))
      finally archive.close()
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("collisions, changed sources, invalid drafts and interruption preserve all existing files") {
    val dir=Files.createTempDirectory("embedding-edit-errors")
    try
      val source=dir.resolve("source.safetensors");val target=dir.resolve("stage.safetensors")
      val original=fixture("edit-source.safetensors",sourceHash);Files.write(source,original)
      Files.writeString(target,"existing")
      intercept[java.nio.file.FileAlreadyExistsException](SafeTensorsEditCopy.build(source,target,edited,sourceHash,Some(baseline)))
      assertEquals(Files.readString(target),"existing")
      Files.delete(target)
      intercept[IllegalArgumentException](SafeTensorsEditCopy.build(source,target,edited,"0"*64,Some(baseline)))
      assert(!Files.exists(target))
      intercept[IllegalArgumentException](SafeTensorsEditCopy.build(source,target,edited.updated("half",Vector(Vector(1.0))),sourceHash,Some(baseline)))
      assert(!Files.exists(target))
      intercept[IllegalArgumentException](SafeTensorsEditCopy.build(source,target,edited.updated("half",Vector(Vector(1.0,-2.0),Vector(.25,1e100))),sourceHash,Some(baseline)))
      assert(!Files.exists(target))
      val halfway=edited.updated("half",Vector(Vector(1.0004882812500002,-2.0),Vector(.25,10.0)))
      SafeTensorsEditCopy.build(source,target,halfway,sourceHash,Some(baseline))
      val half=SafeTensors.open(target)
      try assertEquals(half.readFloat("half").toFloatArray.head,1.0009765625f) finally half.close()
      Files.delete(target)
      var calls=0
      intercept[CancellationException](SafeTensorsEditCopy.build(source,target,edited,sourceHash,Some(baseline),()=>{calls+=1;calls>=3}))
      assert(!Files.exists(target));assertEquals(Files.readAllBytes(source).toVector,original.toVector)
      val displaced=dir.resolve("displaced.safetensors")
      var stageChecks=0
      intercept[CancellationException](SafeTensorsEditCopy.build(source,target,edited,sourceHash,Some(baseline),()=>{
        stageChecks+=1
        if stageChecks==5 then { Files.move(target,displaced);Files.writeString(target,"foreign");true } else false
      }))
      assertEquals(Files.readString(target),"foreign")
      assertEquals(Files.readAllBytes(displaced).toVector,original.toVector)
      val target2=dir.resolve("stage2.safetensors");val displaced2=dir.resolve("displaced2.safetensors")
      var swapChecks=0
      intercept[IllegalArgumentException](SafeTensorsEditCopy.build(source,target2,edited,sourceHash,Some(baseline),()=>{
        swapChecks+=1
        if swapChecks==5 then { Files.move(target2,displaced2);Files.writeString(target2,"other");false } else false
      }))
      assertEquals(Files.readString(target2),"other")
      assertEquals(Files.readAllBytes(displaced2).toVector,original.toVector)
      val files=Files.list(dir)
      try assertEquals(files.count(),5L) finally files.close()
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
