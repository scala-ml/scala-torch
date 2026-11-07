package scalatorch.models.worker

import java.io.InputStream
import java.net.URI
import java.nio.file.{Files,Path,StandardOpenOption}
import java.security.MessageDigest
import scala.util.Using

/** Optional, bounded download of pinned A1111 release weights. */
private[worker] object NativeApproxDownload:
  private val base="https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/"
  private val hashes=Map("model.pt"->"4f88c9078bb2238cdd0d8864671dd33e3f42e091e41f08903f3c15e4a54a9b39",
    "vaeapprox-sdxl.pt"->"6c8db511f910184436a613807ce06a729f4cf96bbf7977c7c6c40d5b3e4a8333")
  def official(target: Path,name: String,cancelled: () => Unit): Unit =
    val hash=hashes.getOrElse(name,throw IllegalArgumentException("unknown Approx NN weights"))
    fetch(target,hash,()=>{
      val connection=URI.create(base+name).toURL.openConnection()
      connection.setConnectTimeout(30000);connection.setReadTimeout(30000)
      connection.getInputStream
    },cancelled)
  def fetch(target: Path,expected: String,open: () => InputStream,cancelled: () => Unit): Unit =
    require(expected.matches("[0-9a-f]{64}"),"invalid weights digest")
    val parent=target.toAbsolutePath.normalize().getParent
    Files.createDirectories(parent)
    if Files.exists(target) then throw new java.nio.file.FileAlreadyExistsException(target.toString)
    val temporary=Files.createTempFile(parent,".approx-download-",".tmp")
    try
      val digest=MessageDigest.getInstance("SHA-256")
      Using.resource(open()) { input =>
        Using.resource(Files.newOutputStream(temporary,StandardOpenOption.WRITE,StandardOpenOption.TRUNCATE_EXISTING)) { output =>
          val buffer=new Array[Byte](65536)
          var size=0L
          var count=input.read(buffer)
          while count != -1 do
            cancelled()
            size+=count
            require(size<=16L*1024*1024,"Approx NN download exceeds size limit")
            digest.update(buffer,0,count);output.write(buffer,0,count)
            count=input.read(buffer)
          require(size>0,"empty Approx NN download")
        }
      }
      require(java.util.HexFormat.of().formatHex(digest.digest())==expected,"Approx NN download checksum mismatch")
      cancelled()
      Files.move(temporary,target)
    finally Files.deleteIfExists(temporary)
