package scalatorch.models.worker

class NativeRgbLabSuite extends munit.FunSuite:
  test("LAB to RGB agrees with OpenCV for every 24-bit LAB value") {
    val reference=sys.env.get("SCALATORCH_EXHAUSTIVE_LAB_RGB")
    assume(reference.nonEmpty,"exhaustive OpenCV fixture required")
    val file=java.nio.file.Files.newInputStream(java.nio.file.Path.of(reference.get))
    try
      val pixels=256*256
      val lab=new Array[Int](pixels*3)
      val expected=new Array[Byte](pixels*3)
      for lightness <- 0 until 256 do
        assertEquals(file.readNBytes(expected,0,expected.length),expected.length)
        for a <- 0 until 256; b <- 0 until 256 do
          val i=(a*256+b)*3
          lab(i)=lightness;lab(i+1)=a;lab(i+2)=b
        val actual=NativeRgbLab.toRgb(lab,()=>false)
        val mismatch=actual.indices.find(i=>actual(i)!=(expected(i)&255))
        assert(mismatch.isEmpty,s"lightness=$lightness index=${mismatch.getOrElse(-1)} actual=${mismatch.map(actual(_))} expected=${mismatch.map(i=>expected(i)&255)}")
      assertEquals(file.read(),-1)
    finally file.close()
  }
  test("RGB to LAB agrees with OpenCV for every 24-bit color") {
    val reference=sys.env.get("SCALATORCH_EXHAUSTIVE_RGB_LAB")
    assume(reference.nonEmpty,"exhaustive OpenCV fixture required")
    val file=java.nio.file.Files.newInputStream(java.nio.file.Path.of(reference.get))
    try
      val pixels=256*256
      val rgb=new Array[Int](pixels*3)
      val expected=new Array[Byte](pixels*3)
      for red <- 0 until 256 do
        val count=file.readNBytes(expected,0,expected.length)
        assertEquals(count,expected.length)
        for green <- 0 until 256; blue <- 0 until 256 do
          val i=(green*256+blue)*3
          rgb(i)=red;rgb(i+1)=green;rgb(i+2)=blue
        val actual=NativeRgbLab.toLab(rgb,()=>false)
        val mismatch=actual.indices.find(i=>actual(i)!=(expected(i)&255))
        assert(mismatch.isEmpty,s"red=$red index=${mismatch.getOrElse(-1)} actual=${mismatch.map(actual(_))} expected=${mismatch.map(i=>expected(i)&255)}")
      assertEquals(file.read(),-1)
    finally file.close()
  }
  test("LAB to RGB agrees with OpenCV on 65536 colors") {
    val stream=this.getClass.getResourceAsStream("/lab-rgb-reference.bin")
    val bytes=try stream.readAllBytes() finally stream.close()
    val half=bytes.length/2
    val actual=NativeRgbLab.toRgb(bytes.take(half).map(_&255),()=>false)
    val expected=bytes.drop(half).map(_&255)
    val differences=actual.indices.filter(i=>actual(i)!=expected(i))
    assert(differences.isEmpty,s"${differences.size} mismatches: ${differences.take(8).map(i=>s"$i:${actual(i)}!=${expected(i)}").mkString(",")}")
  }
  test("RGB to LAB agrees with OpenCV on 65536 colors") {
    val stream=this.getClass.getResourceAsStream("/rgb-lab-reference.bin")
    val bytes=try stream.readAllBytes() finally stream.close()
    val half=bytes.length/2
    val rgb=bytes.take(half).map(_&255)
    val expected=bytes.drop(half).map(_&255)
    val actual=NativeRgbLab.toLab(rgb,()=>false)
    val differences=actual.indices.filter(i=>actual(i)!=expected(i))
    assert(differences.isEmpty,s"${differences.size} mismatches: ${differences.take(8).map(i=>s"$i:${actual(i)}!=${expected(i)}").mkString(",")}")
  }
