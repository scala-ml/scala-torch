package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import NativeJson.*

class NativeInpaintDifferencePreviewSuite extends munit.FunSuite:
  test("preview request matches NUI reference and owns its artifacts") {
    val stream=getClass.getResourceAsStream("/inpaint-difference-preview-reference.json")
    val data=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val w=data("width").asInstanceOf[Number].value.toInt;val h=data("height").asInstanceOf[Number].value.toInt
    val root=Files.createTempDirectory("difference-preview-test")
    try
      def write(key: String): Path =
        val bytes=data(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
        val image=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
        for y <- 0 until h; x <- 0 until w do
          val i=(y*w+x)*3;image.setRGB(x,y,(bytes(i)<<16)|(bytes(i+1)<<8)|bytes(i+2))
        val path=root.resolve(key+".png");ImageIO.write(image,"png",path.toFile);path
      val base=write("base");val altered=write("altered")
      val config=root.resolve("scheduler.json")
      Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(root,config)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      for item <- data("cases").asInstanceOf[Arr].values do
        val fields=item.asInstanceOf[Obj].fields
        val options=fields("options").asInstanceOf[Obj].fields
        val result=backend.execute(NativeRequest("preview","inpaint_difference_preview",options++Map("base_path"->Str(base.toString),"altered_path"->Str(altered.toString),"output_dir"->Str(root.toString)),None),context)
        val dir=Path.of(result("artifact_dir").asInstanceOf[Str].value)
        assertEquals(dir.getParent,root)
        val mask=ImageIO.read(dir.resolve("mask.png").toFile);val preview=ImageIO.read(dir.resolve("preview.png").toFile)
        val expectedMask=fields("mask").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
        val expectedPreview=fields("preview").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
        for y <- 0 until h; x <- 0 until w do
          val i=y*w+x;assertEquals(mask.getRaster.getSample(x,y,0),expectedMask(i))
          val rgb=preview.getRGB(x,y)
          for c <- 0 until 3 do
            val actual=(rgb>>>(16-c*8))&255;val expected=expectedPreview(i*3+c)
            assert(math.abs(actual-expected)<=2,s"$options ($x,$y,$c): $actual != $expected")
        Files.delete(dir.resolve("mask.png"));Files.delete(dir.resolve("preview.png"));Files.delete(dir)
      val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
      intercept[java.util.concurrent.CancellationException] {
        backend.execute(NativeRequest("cancel","inpaint_difference_preview",Map("base_path"->Str(base.toString),"altered_path"->Str(altered.toString),"output_dir"->Str(root.toString)),None),denied)
      }
      val children=Files.list(root)
      try assert(!children.anyMatch(p=>p.getFileName.toString.startsWith("inpaint-difference-"))) finally children.close()
    finally
      val paths=Files.walk(root)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
