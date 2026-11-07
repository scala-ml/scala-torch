package scalatorch.models.worker

import java.nio.{ByteBuffer,ByteOrder}
import scalatorch.*

class NativeVaeLatentSuite extends munit.FunSuite:
  test("CMYK TIFF preserves all four channels for NUI interchange") {
    val dir=java.nio.file.Files.createTempDirectory("vae-latent-cmyk")
    for channels<-Vector(4,16) do
      val stream=getClass.getResourceAsStream(s"/vae-latent/cmyk-$channels.tiff")
      val input=dir.resolve(s"python-$channels.tiff")
      try java.nio.file.Files.copy(stream,input) finally stream.close()
      val count=channels*6
      val values=Array.tabulate(count)(i=>(-12d+24d*i/(count-1)).toFloat)
      val tensor=Tensor.fromFloats(values,Shape.from(Seq(1,channels,2,3)))
      val packed=NativeVaeLatent.packQuantized(tensor)
      val expected=NativeVaeLatent.unpackQuantized(packed.width,packed.height,packed.bytes,channels)
      val decoded=NativeVaeLatent.readCmykTiff(input,channels)
      assertEquals(decoded.toFloatArray.toVector,expected.toFloatArray.toVector)
      val output=dir.resolve(s"native-$channels.tiff")
      NativeVaeLatent.writeCmykTiff(tensor,output)
      assertEquals(NativeVaeLatent.readCmykTiff(output,channels).toFloatArray.toVector,expected.toFloatArray.toVector)
      intercept[java.nio.file.FileAlreadyExistsException](NativeVaeLatent.writeCmykTiff(tensor,output))
  }
  test("safetensors input converts decoder coordinates using stored VAE provenance") {
    import scalatorch.io.{SafeTensorWrite,SafeTensorsWriter}
    val dir=java.nio.file.Files.createTempDirectory("vae-latent-safe")
    val tensor=Tensor.fromFloats(Array(1f,2f,3f,4f),Shape.from(Seq(1,4,1,1)))
    val path=dir.resolve("latent.safetensors")
    SafeTensorsWriter.write(path,Map("tensor"->SafeTensorWrite.F32(tensor)),
      Map("sdfx_vae_tools"->"""{"coordinate":"vae_decoder_input","scaling_factor":0.5,"shift_factor":1.0}"""))
    val result=NativeVaeLatent.readSafetensors(path,4)
    assertEquals(result.toFloatArray.toVector,Vector(0f,0.5f,1f,1.5f))
    val malformed=dir.resolve("bad.safetensors")
    SafeTensorsWriter.write(malformed,Map("tensor"->SafeTensorWrite.F32(tensor)),
      Map("sdfx_vae_tools"->"""{"coordinate":"vae_decoder_input"}"""))
    intercept[IllegalArgumentException](NativeVaeLatent.readSafetensors(malformed,4))
    intercept[IllegalArgumentException](NativeVaeLatent.readSafetensors(path,16))
  }
  test("lossless grayscale latent PNG preserves Float32 bits and checks channels") {
    val dir=java.nio.file.Files.createTempDirectory("vae-latent-l")
    for channels<-Vector(4,16) do
      val values=Array.tabulate(channels*6)(i=>if i==0 then -0f else (i-17).toFloat/13f)
      val tensor=Tensor.fromFloats(values,Shape.from(Seq(1,channels,2,3)))
      val reference=dir.resolve(s"reference-$channels.png")
      val input=this.getClass.getResourceAsStream(s"/vae-latent/l-$channels.png")
      try java.nio.file.Files.copy(input,reference) finally input.close()
      val pythonResult=NativeVaeLatent.readExactPng(reference,channels)
      assertEquals(pythonResult.toFloatArray.map(java.lang.Float.floatToRawIntBits).toVector,
        values.map(java.lang.Float.floatToRawIntBits).toVector)
      val path=dir.resolve(s"native-$channels.png")
      NativeVaeLatent.writeExactPng(tensor,path)
      val result=NativeVaeLatent.readExactPng(path,channels)
      assertEquals(result.shape,tensor.shape)
      assertEquals(result.toFloatArray.map(java.lang.Float.floatToRawIntBits).toVector,
        values.map(java.lang.Float.floatToRawIntBits).toVector)
      intercept[IllegalArgumentException](NativeVaeLatent.readExactPng(path,if channels==4 then 16 else 4))
      intercept[java.nio.file.FileAlreadyExistsException](NativeVaeLatent.writeExactPng(tensor,path))
    val rgb=new java.awt.image.BufferedImage(2,2,java.awt.image.BufferedImage.TYPE_INT_RGB)
    val invalid=dir.resolve("rgb.png")
    javax.imageio.ImageIO.write(rgb,"png",invalid.toFile)
    intercept[IllegalArgumentException](NativeVaeLatent.readExactPng(invalid,4))
  }
  test("RGBA PNG latent interchange preserves all channels including transparent pixels") {
    val dir=java.nio.file.Files.createTempDirectory("vae-latent-png")
    for channels<-Vector(4,16) do
      val fixture=this.getClass.getResourceAsStream(s"/vae-latent/rgba-$channels.png")
      val input=dir.resolve(s"reference-$channels.png")
      try java.nio.file.Files.copy(fixture,input) finally fixture.close()
      val expectedStream=this.getClass.getResourceAsStream(s"/vae-latent/rgba-$channels.f32")
      val raw=try expectedStream.readAllBytes() finally expectedStream.close()
      val buffer=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
      val expected=Array.fill(raw.length/4)(buffer.getFloat())
      val decoded=NativeVaeLatent.readRgbaPng(input,channels)
      assertEquals(decoded.shape,Shape.from(Seq(1,channels,2,3)))
      assertEquals(decoded.toFloatArray.toVector,expected.toVector)
      val output=dir.resolve(s"native-$channels.png")
      NativeVaeLatent.writeRgbaPng(decoded,output)
      assertEquals(NativeVaeLatent.readRgbaPng(output,channels).toFloatArray.toVector,expected.toVector)
      intercept[java.nio.file.FileAlreadyExistsException](NativeVaeLatent.writeRgbaPng(decoded,output))
    val transparent=Tensor.fromFloats(Array(10f,-10f,3f,-10f),Shape.from(Seq(1,4,1,1)))
    val transparentPath=dir.resolve("transparent.png")
    val packed=NativeVaeLatent.writeRgbaPng(transparent,transparentPath)
    assertEquals(packed.bytes.toVector.map(_&255),Vector(255,0,166,0))
    assertEquals(NativeVaeLatent.readRgbaPng(transparentPath,4).toFloatArray.toVector,
      NativeVaeLatent.unpackQuantized(1,1,packed.bytes,4).toFloatArray.toVector)
    val rgb=new java.awt.image.BufferedImage(2,2,java.awt.image.BufferedImage.TYPE_INT_RGB)
    val invalid=dir.resolve("rgb.png")
    javax.imageio.ImageIO.write(rgb,"png",invalid.toFile)
    intercept[IllegalArgumentException](NativeVaeLatent.readRgbaPng(invalid,4))
    val oversized=ByteBuffer.allocate(29).order(ByteOrder.BIG_ENDIAN)
    oversized.put(Array[Byte](137.toByte,80,78,71,13,10,26,10)).putInt(13).putInt(0x49484452)
      .putInt(100000).putInt(100000).put(Array[Byte](8,6,0,0,0))
    val bomb=dir.resolve("oversized.png")
    java.nio.file.Files.write(bomb,oversized.array())
    intercept[IllegalArgumentException](NativeVaeLatent.readRgbaPng(bomb,4))
  }
  test("quantized four- and sixteen-channel layouts match NUI RGBA and CMYK fixtures") {
    def bytes(name: String): Array[Byte] =
      val input=this.getClass.getResourceAsStream(s"/vae-latent/$name")
      try input.readAllBytes() finally input.close()
    for channels<-Vector(4,16); mode<-Vector("rgba","cmyk") do
      val count=channels*2*3
      val values=Array.tabulate(count)(i=>(-12d+24d*i/(count-1)).toFloat)
      val source=Tensor.fromFloats(values,Shape.from(Seq(1,channels,2,3)))
      val packed=NativeVaeLatent.packQuantized(source)
      assertEquals((packed.width,packed.height),if channels==4 then (3,2) else (6,4))
      assertEquals(packed.clipped,if channels==4 then 4 else 16)
      assertEquals(packed.bytes.toVector,bytes(s"$mode-$channels.bytes").toVector)
      val raw=bytes(s"$mode-$channels.f32")
      val buffer=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
      val expected=Array.fill(raw.length/4)(buffer.getFloat())
      val decoded=NativeVaeLatent.unpackQuantized(packed.width,packed.height,packed.bytes,channels)
      assertEquals(decoded.shape,source.shape)
      decoded.toFloatArray.zip(expected).foreach((a,b)=>assert(math.abs(a-b)<1e-6f,s"$a != $b"))
  }
  test("latent packing rejects nonfinite, wrong-channel and malformed layouts") {
    intercept[IllegalArgumentException](NativeVaeLatent.packQuantized(
      Tensor.fromFloats(Array(Float.NaN,0f,0f,0f),Shape.from(Seq(1,4,1,1)))))
    intercept[IllegalArgumentException](NativeVaeLatent.unpackQuantized(3,3,new Array[Byte](36),16))
    intercept[IllegalArgumentException](NativeVaeLatent.unpackQuantized(2,2,new Array[Byte](15),4))
    intercept[IllegalArgumentException](NativeVaeLatent.unpackQuantized(2,2,new Array[Byte](16),8))
  }
