package scalatorch.models.worker

import java.nio.file.{Files,Path,SimpleFileVisitor,FileVisitResult}
import java.nio.file.attribute.BasicFileAttributes
import java.util.Locale
import scala.collection.mutable

/** File discovery does not imply that the native engine can execute the associated feature. */
object NativeFileCatalog:
  import NativeJson.*
  private val kinds=Set("vae","lora","controlnet","embeddings","upscalers","face_restore","svd","zero123","photomaker")
  private val extensions=Set(".safetensors",".pt",".pth",".ckpt",".bin",".patch")
  private val markers=Map("controlnet"->"config.json","svd"->"model_index.json","zero123"->"model_index.json")
  def scan(kind: String,dirs: Vector[String],checkCancelled: () => Unit): Vector[Obj]=
    require(kinds(kind),s"unknown file kind: $kind")
    val seen=mutable.Set.empty[Path]
    val entries=mutable.ArrayBuffer.empty[(String,Obj)]
    var visited=0
    def check(): Unit=
      checkCancelled()
      visited+=1
      require(visited<=1000000,"file catalog traversal exceeds limit")
    def size(directory: Path): BigInt=
      var bytes=BigInt(0)
      Files.walkFileTree(directory,new SimpleFileVisitor[Path]:
        override def preVisitDirectory(p: Path,a: BasicFileAttributes): FileVisitResult=
          check();FileVisitResult.CONTINUE
        override def visitFile(p: Path,a: BasicFileAttributes): FileVisitResult=
          check()
          if a.isRegularFile then bytes+=a.size()
          FileVisitResult.CONTINUE
        override def visitFileFailed(p: Path,e: java.io.IOException): FileVisitResult=FileVisitResult.CONTINUE
      )
      bytes
    for raw <- dirs if raw.nonEmpty do
      val base=Path.of(raw).toAbsolutePath.normalize()
      if Files.isDirectory(base) then
        def add(path: Path,directory: Boolean): Unit=
          val canonical=path.toRealPath()
          if seen.add(canonical) then
            require(entries.size<100000,"file catalog result exceeds limit")
            val relative=base.relativize(path).toString.replace('\\','/')
            val name=if directory && relative.isEmpty then canonical.getFileName.toString
              else if directory then relative else relative.substring(0,relative.lastIndexOf('.'))
            val bytes=if directory then size(path) else BigInt(Files.size(path))
            entries+=name->Obj(Map("name"->Str(name),"path"->Str(path.toString),
              "size_mb"->Number((BigDecimal(bytes)/1048576).setScale(2,BigDecimal.RoundingMode.HALF_EVEN))))
        Files.walkFileTree(base,new SimpleFileVisitor[Path]:
          override def preVisitDirectory(path: Path,a: BasicFileAttributes): FileVisitResult=
            check()
            if markers.get(kind).exists(marker => Files.isRegularFile(path.resolve(marker))) then
              add(path,true);FileVisitResult.SKIP_SUBTREE
            else if path!=base && path.getFileName.toString.startsWith(".") then FileVisitResult.SKIP_SUBTREE
            else FileVisitResult.CONTINUE
          override def visitFile(path: Path,a: BasicFileAttributes): FileVisitResult=
            check()
            val name=path.getFileName.toString.toLowerCase(Locale.ROOT)
            if a.isRegularFile && extensions.exists(name.endsWith) then add(path,false)
            FileVisitResult.CONTINUE
          override def visitFileFailed(p: Path,e: java.io.IOException): FileVisitResult=FileVisitResult.CONTINUE
        )
    entries.toVector.sortBy((name,_) => (name.toLowerCase(Locale.ROOT),name)).map(_._2)
