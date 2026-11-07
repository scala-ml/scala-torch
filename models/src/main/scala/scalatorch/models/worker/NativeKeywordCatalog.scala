package scalatorch.models.worker

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files,Path,FileAlreadyExistsException,StandardOpenOption}

private[worker] object NativeKeywordCatalog:
  private val url="https://raw.githubusercontent.com/mix1009/model-keyword/c3d0833a9434bb41ecd7ea715b5abd5e6b43a5af/lora-keyword.txt"
  private val expectedHash="b20edcfd8f121c6a5ce9012e4be7ae587a4a66b85f2cd0328768523df78db5ec"
  private val expectedSize=31395702L

  def download(directory: Path,confirmed: Boolean,context: NativeWorkContext): Path =
    install(directory,confirmed,() => {
      val connection=java.net.URI.create(url).toURL.openConnection()
      connection.setConnectTimeout(30000)
      connection.setReadTimeout(30000)
      connection.getInputStream
    },expectedSize,expectedHash,context)

  def install(directory: Path,confirmed: Boolean,open: () => InputStream,expectedSize: Long,
      expectedHash: String,context: NativeWorkContext): Path =
    require(confirmed,"explicit confirmation is required to download the catalog")
    require(expectedSize>=0 && expectedHash.matches("[0-9a-f]{64}"),"invalid pinned catalog identity")
    context.checkCancelled()
    val root=directory.toAbsolutePath.normalize()
    Files.createDirectories(root)
    val destination=root.resolve("lora-keyword.txt")
    if Files.exists(destination) then throw new FileAlreadyExistsException(destination.toString)
    val temporary=Files.createTempFile(root,".model-keyword-download-",".tmp")
    try
      val digest=java.security.MessageDigest.getInstance("SHA-256")
      var count=0L
      scala.util.Using.resource(FileChannel.open(temporary,StandardOpenOption.WRITE,StandardOpenOption.TRUNCATE_EXISTING)) { output =>
        scala.util.Using.resource(open()) { source =>
          val buffer=new Array[Byte](65536)
          var done=false
          while !done do
            context.checkCancelled()
            val size=source.read(buffer)
            if size<0 then done=true
            else if size>0 then
              count+=size
              require(count<=expectedSize,"download exceeds the pinned catalog size")
              digest.update(buffer,0,size)
              val data=ByteBuffer.wrap(buffer,0,size)
              while data.hasRemaining do
                context.checkCancelled()
                output.write(data)
        }
        require(count==expectedSize && java.util.HexFormat.of().formatHex(digest.digest())==expectedHash,
          "download does not match the pinned catalog hash and size")
        output.force(true)
      }
      context.beginPublication()
      Files.createLink(destination,temporary)
      destination
    finally Files.deleteIfExists(temporary)
