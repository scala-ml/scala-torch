package scalatorch.models.worker

import java.nio.file.{Files,Path,SimpleFileVisitor,FileVisitResult}
import java.nio.file.attribute.BasicFileAttributes
import java.util.Locale
import scala.collection.mutable
import NativeJson.*

/** Discovery describes files, not a guarantee that every architecture is supported. */
object NativeModelCatalog:
  def scan(dirs: Vector[String],checkCancelled: () => Unit): Vector[Obj] =
    val seen=mutable.Set.empty[Path]
    val entries=mutable.ArrayBuffer.empty[Obj]
    var visits=0
    def check(): Unit =
      checkCancelled()
      visits+=1
      require(visits<=1000000,"model catalog traversal exceeds limit")
    def pipeline(path: Path): Boolean =
      Files.isRegularFile(path.resolve("unet/config.json")) &&
        Vector("vae","text_encoder","tokenizer").forall(x=>Files.isDirectory(path.resolve(x)))
    def size(path: Path): BigInt =
      var total=BigInt(0)
      Files.walkFileTree(path,new SimpleFileVisitor[Path]:
        override def preVisitDirectory(p: Path,a: BasicFileAttributes): FileVisitResult = { check();FileVisitResult.CONTINUE }
        override def visitFile(p: Path,a: BasicFileAttributes): FileVisitResult =
          check()
          if a.isRegularFile then total+=a.size()
          FileVisitResult.CONTINUE
        override def visitFileFailed(p: Path,e: java.io.IOException): FileVisitResult = FileVisitResult.CONTINUE
      )
      total
    def add(path: Path,isDirectory: Boolean): Unit =
      val real=path.toRealPath()
      if seen.add(real) then
        require(entries.size<100000,"model catalog result exceeds limit")
        val bytes=if isDirectory then size(real) else BigInt(Files.size(real))
        entries+=Obj(Map("name"->Str(real.getFileName.toString),"path"->Str(real.toString),
          "kind"->Str(if isDirectory then "diffusers" else if real.getFileName.toString.toLowerCase(Locale.ROOT).endsWith(".ckpt") then "ckpt" else "safetensors"),
          "size_mb"->Number(BigDecimal(bytes)/BigDecimal(1048576))))
    dirs.filter(_.nonEmpty).foreach { raw =>
      val base=Path.of(raw).toAbsolutePath.normalize()
      if Files.isDirectory(base) then Files.walkFileTree(base,new SimpleFileVisitor[Path]:
        override def preVisitDirectory(p: Path,a: BasicFileAttributes): FileVisitResult =
          check()
          if pipeline(p) then { add(p,true);FileVisitResult.SKIP_SUBTREE }
          else if p!=base && p.getFileName.toString.startsWith(".") then FileVisitResult.SKIP_SUBTREE
          else FileVisitResult.CONTINUE
        override def visitFile(p: Path,a: BasicFileAttributes): FileVisitResult =
          check()
          if a.isRegularFile && Vector(".safetensors",".ckpt").exists(p.getFileName.toString.toLowerCase(Locale.ROOT).endsWith) then add(p,false)
          FileVisitResult.CONTINUE
        override def visitFileFailed(p: Path,e: java.io.IOException): FileVisitResult = FileVisitResult.CONTINUE
      )
    }
    entries.toVector.sortBy(e=>e.fields("name").asInstanceOf[Str].value.toLowerCase(Locale.ROOT))
