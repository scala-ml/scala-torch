package scalatorch.models.worker

import scalatorch.models.DiffusionImageInput
import java.nio.file.{Files,Path}
import java.util.UUID
import javax.imageio.{IIOImage,ImageIO}
import javax.imageio.metadata.IIOMetadataNode
import NativeJson.*
import scala.util.control.NonFatal

/** Standalone ControlNet preprocessing, independent of model loading. */
private[worker] object NativePreprocess:
  private val valid=Set("none","inpaint","invert","canny","recolor_luminance","recolor_intensity",
    "tile_resample","tile_colorfix","blur_gaussian","depth","depth_marigold","normal_marigold",
    "openpose","lineart","lineart_anime","softedge_hed","mlsd","normal_bae")
  private val thresholdKeys=Map("canny"->("low_threshold",100.0,"high_threshold",200.0),
    "recolor_luminance"->("gamma_correction",1.0,"",0.0),"recolor_intensity"->("gamma_correction",1.0,"",0.0),
    "tile_resample"->("down_sampling_rate",1.0,"",0.0),"tile_colorfix"->("variation",8.0,"",0.0),
    "blur_gaussian"->("sigma",9.0,"",0.0))
  private[worker] def sourceParameters(path: Path,key: String="parameters"): Option[String] =
    try
      val in=ImageIO.createImageInputStream(path.toFile)
      if in==null then None else
        try
          val readers=ImageIO.getImageReaders(in)
          if !readers.hasNext then None else
            val reader=readers.next()
            try
              reader.setInput(in,false,false)
              val meta=reader.getImageMetadata(0)
              if !meta.getMetadataFormatNames.contains("javax_imageio_png_1.0") then None else
                def find(node: org.w3c.dom.Node): Option[String] =
                  val attrs=node.getAttributes
                  if attrs!=null && attrs.getNamedItem("keyword")!=null && attrs.getNamedItem("keyword").getNodeValue==key then
                    Option(attrs.getNamedItem("value")).orElse(Option(attrs.getNamedItem("text"))).map(_.getNodeValue)
                  else
                    val children=node.getChildNodes
                    (0 until children.getLength).iterator.map(i=>find(children.item(i))).collectFirst { case Some(s) => s }
                find(meta.getAsTree("javax_imageio_png_1.0"))
            finally reader.dispose()
        finally in.close()
    catch case NonFatal(_) => None
  private[worker] def write(image: java.awt.image.BufferedImage,path: Path,texts: Vector[(String,String)]): Unit =
    val writer=ImageIO.getImageWritersByFormatName("png").next()
    val out=ImageIO.createImageOutputStream(path.toFile)
    try
      writer.setOutput(out)
      val metadata=writer.getDefaultImageMetadata(javax.imageio.ImageTypeSpecifier.createFromRenderedImage(image),writer.getDefaultWriteParam)
      if texts.nonEmpty then
        val root=new IIOMetadataNode("javax_imageio_png_1.0")
        val entries=new IIOMetadataNode("iTXt")
        texts.foreach { (key,value) =>
          val node=new IIOMetadataNode("iTXtEntry")
          node.setAttribute("keyword",key);node.setAttribute("compressionFlag","FALSE");node.setAttribute("compressionMethod","0")
          node.setAttribute("languageTag","");node.setAttribute("translatedKeyword","");node.setAttribute("text",value)
          entries.appendChild(node)
        }
        root.appendChild(entries);metadata.mergeTree("javax_imageio_png_1.0",root)
      writer.write(null,new IIOImage(image,null,metadata),writer.getDefaultWriteParam)
    finally { writer.dispose();out.close() }
  def execute(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    require(p.keySet.subsetOf(Set("preprocessor","image_path","output_dir","args","standalone","processor_res","pixel_perfect")),
      "unknown preprocess parameters")
    def string(key: String,default: String): String=p.get(key) match
      case None => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    def bool(key: String): Boolean=p.get(key) match
      case None => false
      case Some(Bool(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be boolean")
    val name=string("preprocessor","none")
    if !valid(name) then throw new NativeBackendError("unsupported",s"preprocessor '$name' is not available")
    val source=string("image_path","");val destination=string("output_dir","")
    require(source.nonEmpty && destination.nonEmpty,"image_path and output_dir are required")
    val args=p.get("args") match
      case None => Map.empty[String,NativeJson]
      case Some(Obj(values)) =>
        require(values.values.forall(_.isInstanceOf[Number]),"preprocessor args must be numeric");values
      case _ => throw IllegalArgumentException("args must be an object")
    val standalone=bool("standalone");val pixelPerfect=bool("pixel_perfect")
    val requested=p.get("processor_res") match
      case None => -1
      case Some(Number(n)) if n.isValidInt => n.toInt
      case _ => throw IllegalArgumentException("processor_res must be an integer")
    val image=DiffusionImageInput.readRgb(Path.of(source))
    val resolution=if name!="canny" then -1 else if pixelPerfect then math.min(image.getWidth,image.getHeight)
      else if requested>=64 && requested<=2048 then requested else 512
    val notes=if standalone && name!="canny" && (pixelPerfect || requested!= -1) then
      Vector(s"preprocessor '$name' has no resolution; processor_res and pixel_perfect are ignored") else Vector.empty
    val output=NativeControlNet.preprocessImage(name,image,args,standalone,resolution,context.cancelled)
    val (keyA,defaultA,keyB,defaultB)=thresholdKeys.getOrElse(name,("",0.0,"",0.0))
    def threshold(key: String,default: Double): NativeJson=args.get(key).getOrElse(Number(BigDecimal(default)))
    val a=threshold(keyA,defaultA);val b=threshold(keyB,defaultB)
    val info=Obj(Map("module"->Str(name),"processor_res"->Number(resolution),"threshold_a"->a,"threshold_b"->b))
    def rendered(key: String,value: NativeJson): String =
      val n=value.asInstanceOf[Number].value
      if args.contains(key) || Set("gamma_correction","down_sampling_rate","sigma")(key) then n.toDouble.toString
      else n.bigDecimal.stripTrailingZeros.toPlainString
    val infotext=s"ControlNet Preprocessor: \"{'module': '$name', 'processor_res': $resolution, 'threshold_a': ${rendered(keyA,a)}, 'threshold_b': ${rendered(keyB,b)}}\", Size: ${output.getWidth}x${output.getHeight}"
    val outputDir=Path.of(destination).toAbsolutePath.normalize();Files.createDirectories(outputDir)
    val temp=Files.createTempFile(outputDir,".native-preprocess-",".png")
    try
      val texts=if standalone then Vector("parameters"->infotext)++sourceParameters(Path.of(source)).filter(_.nonEmpty).map("Source parameters"->_)
        else Vector.empty
      write(output,temp,texts)
      context.checkCancelled()
      val target=outputDir.resolve(s"native-preprocess-${UUID.randomUUID()}.png")
      context.beginPublication();Files.move(temp,target)
      notes.foreach(note=>context.event("log",Map("level"->Str("warn"),"message"->Str(note))))
      if standalone then Map("path"->Str(target.toString),"info"->info,"notes"->Arr(notes.map(Str.apply)))
      else Map("path"->Str(target.toString))
    finally Files.deleteIfExists(temp)
