package scalatorch.models.worker

import java.nio.file.{Files,Path,StandardOpenOption}
import java.util.UUID
import java.util.concurrent.CancellationException
import javax.imageio.ImageIO
import scala.util.control.NonFatal
import scalatorch.models.DiffusionImageInput
import scalatorch.io.{SafeTensorWrite,SafeTensorsWriter}
import NativeJson.*

/** Initial VAE Tools request implementation; unsupported variants fail explicitly. */
private[worker] object NativeVaeTools:
  def execute(p: Map[String,NativeJson],tool: NativeVaeToolsCodec,context: NativeWorkContext,
      provenance: Map[String,NativeJson]=Map.empty,password: Option[String]=None): Map[String,NativeJson] =
    def addPngMetadata(file: Path,fields: Map[String,NativeJson]): Unit =
      val json=NativeJson.render(Obj(fields)).getBytes(java.nio.charset.StandardCharsets.UTF_8)
      require(json.length<=1048576,"PNG metadata exceeds limit")
      val keyword="VAE Tools".getBytes(java.nio.charset.StandardCharsets.ISO_8859_1)
      val payload=keyword++Array.fill[Byte](5)(0)++json
      val kind=Array[Byte](105,84,88,116)
      val crc=new java.util.zip.CRC32()
      crc.update(kind);crc.update(payload)
      val fileHandle=new java.io.RandomAccessFile(file.toFile,"rw")
      try
        val length=fileHandle.length()
        require(length>=45,"invalid PNG")
        fileHandle.seek(0)
        val header=new Array[Byte](16);fileHandle.readFully(header)
        require(header.sameElements(Array[Byte](-119,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82)),"invalid PNG header")
        val offset=33L
        val extra=payload.length+12L
        val buffer=new Array[Byte](65536)
        var end=length
        while end>offset do
          val size=math.min(buffer.length.toLong,end-offset).toInt
          val start=end-size
          fileHandle.seek(start);fileHandle.readFully(buffer,0,size)
          fileHandle.seek(start+extra);fileHandle.write(buffer,0,size)
          end=start
        fileHandle.seek(offset)
        fileHandle.writeInt(payload.length);fileHandle.write(kind);fileHandle.write(payload)
        fileHandle.writeInt(crc.getValue.toInt)
      finally fileHandle.close()
    def string(key: String,default: String): String=p.get(key) match
      case None | Some(Null) => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    def bool(key: String,default: Boolean): Boolean=p.get(key) match
      case None => default
      case Some(Bool(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be boolean")
    val operation=string("operation","encode")
    val representation=string("representation","L")
    val approximation=string("approximation","Full")
    require(Set("encode","decode","encode+decode")(operation),"invalid VAE operation")
    require(Set("L","RGBA","CMYK")(representation),"invalid latent representation")
    val tiled=bool("tiled",false)
    if !(approximation=="TAESD" && !tiled || Set("Full","Approx cheap","Approx NN")(approximation)) then
      throw new NativeBackendError("unsupported","native VAE Tools approximation is not available")
    val cmykFormat=string("cmyk_format","TIFF")
    require(Set("TIFF","JPEG100")(cmykFormat),"invalid CMYK format")
    if representation=="CMYK" && cmykFormat!="TIFF" then
      throw new NativeBackendError("unsupported","CMYK JPEG output is not available with the JDK image codecs")
    bool("allow_download",false)
    val consent=bool("allow_plaintext_latent",false)
    require(password.isEmpty || operation!="encode" || consent,"VAE encode saves readable latent data; plaintext consent required")
    val seed=p.getOrElse("seed",Number(0)) match
      case Number(n) if n.isWhole && n>=0 && n<=BigDecimal("4294967295") => n
      case _ => throw IllegalArgumentException("invalid seed")
    val inputs=p.get("input_paths") match
      case Some(Arr(values)) if values.nonEmpty && values.size<=64 => values.map {
        case Str(value) if value.nonEmpty => Path.of(value)
        case _ => throw IllegalArgumentException("invalid input path")
      }
      case _ => throw IllegalArgumentException("input_paths must contain 1..64 paths")
    val parent=Path.of(string("output_dir",".")).toAbsolutePath.normalize().resolve("vae-tools")
    Files.createDirectories(parent)
    val destination=parent.resolve(s"run-${UUID.randomUUID()}")
    val stage=Files.createTempDirectory(parent,".native-vae-")
    var published=false
    var legacyUnverified=false
    def path(name: String)=stage.resolve(name)
    def finalPath(name: String)=destination.resolve(name).toString
    def png(image: java.awt.image.BufferedImage,name: String): Unit =
      password match
        case Some(pass) => NativeProtectedPng.write(image,path(name),pass,Vector.empty)
        case None =>
          val output=Files.newOutputStream(path(name),StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
          try require(ImageIO.write(image,"png",output),"PNG writer unavailable") finally output.close()
    def pngInput(file: Path): Boolean =
      if !Files.isRegularFile(file) then false
      else scala.util.Using.resource(Files.newInputStream(file))(_.readNBytes(8)).sameElements(Array[Byte](-119,80,78,71,13,10,26,10))
    def jpegInput(file: Path): Boolean =
      if !Files.isRegularFile(file) then false
      else scala.util.Using.resource(Files.newInputStream(file))(_.readNBytes(2)).sameElements(Array[Byte](-1,-40))
    def readRgba(file: Path): scalatorch.Tensor[Float] =
      val decoded=NativeProtectedPng.read(file,password)
      if decoded.scrambled && decoded.verified.isEmpty then legacyUnverified=true
      if !decoded.scrambled then NativeVaeLatent.readRgbaPng(file,4)
      else
        val image=decoded.image;val width=image.getWidth;val height=image.getHeight
        require(image.getRaster.getNumBands==4 && image.getColorModel.hasAlpha &&
          image.getSampleModel.getSampleSize.forall(_==8) && !image.isAlphaPremultiplied,"latent image must be 8-bit RGBA")
        val data=image.getRaster.getPixels(0,0,width,height,null.asInstanceOf[Array[Int]]).map(_.toByte)
        NativeVaeLatent.unpackQuantized(width,height,data,4)
    try
      val results=inputs.zipWithIndex.map { (input,index) =>
        context.checkCancelled()
        val start=System.nanoTime()
        val prefix=f"${index+1}%03d"
        val item=try
          var rawCoordinates=false
          val latent=if operation=="decode" then
            if input.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
              val (value,raw)=NativeVaeLatent.readSafetensorsWithCoordinate(input,4,approximation=="Full")
              rawCoordinates=raw
              value
            else if representation=="L" then
              if NativePreprocess.sourceParameters(input,"Encrypt").nonEmpty then
                val protectedImage=NativeProtectedPng.read(input,password,67174400L)
                if protectedImage.scrambled && protectedImage.verified.isEmpty then legacyUnverified=true
                NativeVaeLatent.readExactImage(protectedImage.image,4)
              else NativeVaeLatent.readExactPng(input,4)
            else if representation=="RGBA" then readRgba(input)
            else NativeVaeLatent.readCmykTiff(input,4)
          else
            val image=if pngInput(input) then
              val decoded=NativeProtectedPng.read(input,password)
              if decoded.scrambled && decoded.verified.isEmpty then legacyUnverified=true
              decoded.image
            else DiffusionImageInput.readRgb(input,compositeAlpha=false,applyOrientation= !jpegInput(input))
            tool.encodeSeeded(image,seed.toLong+index,()=>context.checkCancelled())
          val encoded=s"$prefix-encoded.${if representation=="CMYK" then "tiff" else "png"}"
          val raw=s"$prefix-latent.safetensors"
          if operation=="encode" then
            if representation=="L" then NativeVaeLatent.writeExactPng(latent,path(encoded))
            else if representation=="RGBA" && password.nonEmpty then
              val packed=NativeVaeLatent.packQuantized(latent)
              val image=new java.awt.image.BufferedImage(packed.width,packed.height,java.awt.image.BufferedImage.TYPE_4BYTE_ABGR)
              image.getRaster.setPixels(0,0,packed.width,packed.height,packed.bytes.map(_&255))
              png(image,encoded)
            else if representation=="RGBA" then NativeVaeLatent.writeRgbaPng(latent,path(encoded))
            else NativeVaeLatent.writeCmykTiff(latent,path(encoded))
          val restored=if operation=="encode+decode" then
            if representation=="L" then latent
            else
              val packed=NativeVaeLatent.packQuantized(latent)
              NativeVaeLatent.unpackQuantized(packed.width,packed.height,packed.bytes,4)
          else latent
          val dims=latent.shape.dims
          val stats=Obj(Map("shape"->Arr(dims.map(n=>Number(BigDecimal(n))).toVector),
            "quantized"->Bool(representation!="L")))
          val details=provenance++Map[String,NativeJson]("coordinate"->Str("diffusion_scaled"),
            "operation"->Str(operation),"representation"->Str(representation),"approximation"->Str(approximation),
            "seed"->Number(seed+index),"statistics"->stats)
          if operation=="encode" && representation!="CMYK" then addPngMetadata(path(encoded),details)
          val resultName=if operation=="encode" then encoded else s"$prefix-decoded.png"
          val display=if operation=="encode" then
            val metadata=NativeJson.render(Obj(provenance.updated("coordinate",Str("diffusion_scaled"))))
            SafeTensorsWriter.write(path(raw),Map("tensor"->SafeTensorWrite.F32(latent)),
              metadata=Map("sdfx_vae_tools"->metadata),cancelled=context.cancelled)
            if representation=="RGBA" && password.nonEmpty then NativeProtectedPng.read(path(encoded),password).image
            else if representation!="CMYK" then ImageIO.read(path(encoded).toFile)
            else
              val packed=NativeVaeLatent.packQuantized(latent)
              val image=new java.awt.image.BufferedImage(packed.width,packed.height,java.awt.image.BufferedImage.TYPE_INT_RGB)
              for y<-0 until packed.height; x<-0 until packed.width do
                val i=(y*packed.width+x)*4
                val k=packed.bytes(i+3)&255
                def rgb(c: Int)=((255-(packed.bytes(i+c)&255))*(255-k)+127)/255
                image.setRGB(x,y,(rgb(0)<<16)|(rgb(1)<<8)|rgb(2))
              image
          else
            val image=if rawCoordinates then tool.decodeRaw(restored,()=>context.checkCancelled())
              else tool.decode(restored,()=>context.checkCancelled())
            png(image,resultName)
            addPngMetadata(path(resultName),details)
            image
          val preview=s"$prefix-preview.png"
          val previewImage=if operation=="encode" then
            val rgb=new java.awt.image.BufferedImage(display.getWidth,display.getHeight,java.awt.image.BufferedImage.TYPE_INT_RGB)
            for y<-0 until display.getHeight; x<-0 until display.getWidth do rgb.setRGB(x,y,display.getRGB(x,y))
            rgb
          else display
          png(previewImage,preview)
          addPngMetadata(path(preview),details.updated("display_only",Bool(true)))
          Map[String,NativeJson]("input_path"->Str(input.toString),"status"->Str("complete"),
            "path"->Str(finalPath(resultName)),"preview_path"->Str(finalPath(preview)),
            "width"->Number(display.getWidth),"height"->Number(display.getHeight),
            "statistics"->stats)++
            (if operation=="encode" then Map("latent_path"->Str(finalPath(raw))) else Map.empty)
        catch
          case e: CancellationException => throw e
          case NonFatal(e) =>
            val files=Files.list(stage)
            try files.filter(p=>p.getFileName.toString.startsWith(s"$prefix-")).forEach(Files.deleteIfExists(_)) finally files.close()
            Map[String,NativeJson]("input_path"->Str(input.toString),"status"->Str("error"),
              "code"->Str("invalid_params"),"error"->Str(Option(e.getMessage).getOrElse("invalid input")))
        context.event("progress",Map("step"->Number(index+1),"total_steps"->Number(inputs.size),"phase"->Str("vae_tools")))
        Obj(item.updated("elapsed_seconds",Number(BigDecimal((System.nanoTime()-start)/1e9))))
      }
      val encoding=if Set("Approx cheap","Approx NN")(approximation) && operation!="decode" then "Full" else approximation
      val warnings=(if legacyUnverified then Vector(Str("Legacy image password is unverified; decoded pixels may be incorrect.")) else Vector.empty)++
        (if password.nonEmpty && operation=="encode" then Vector(Str("Latent safetensors and report metadata remain readable plaintext."))
        else if password.nonEmpty then Vector(Str("Report metadata remains readable plaintext.")) else Vector.empty)++
        (if Set("Approx cheap","Approx NN")(approximation) && operation!="decode" then
        Vector(Str(s"$approximation cannot encode; Full is used for encoding")) else Vector.empty)++
        (if encoding=="Full" && operation!="decode" then
          Vector(Str("Full VAE sampling uses the PyTorch CPU random stream; CUDA PyTorch seeds can produce different samples.")) else Vector.empty)
      val report=Map[String,NativeJson]("operation"->Str(operation),"representation"->Str(representation),
        "approximation"->Str(approximation),"encoding_approximation"->Str(encoding),"tiled"->Bool(tiled),
        "seed"->Number(seed),"channels"->Number(4),"warnings"->Arr(warnings),"results"->Arr(results),
        "report_path"->Str(finalPath("report.json")))
      Files.writeString(path("report.json"),NativeJson.render(Obj(report)),StandardOpenOption.CREATE_NEW)
      context.checkCancelled()
      context.beginPublication()
      Files.move(stage,destination)
      published=true
      report
    finally
      if !published then
        val files=Files.walk(stage)
        try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally files.close()
