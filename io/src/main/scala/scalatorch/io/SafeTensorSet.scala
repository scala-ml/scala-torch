package scalatorch.io

import scalatorch.Tensor
import java.nio.file.{Files, LinkOption, Path}
import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal

/** Owns validated single-file or indexed weights. Do not modify files while this set is open. */
final class SafeTensorSet private (
    archives: Vector[SafeTensors], owners: Map[String,SafeTensors], val tensors: Map[String,TensorInfo]
) extends AutoCloseable:
  private var closed = false
  private def owner(name: String): SafeTensors =
    if closed then throw IllegalStateException("safetensors set is closed")
    owners.getOrElse(name,throw IllegalArgumentException(s"tensor not found: $name"))
  def readFloat(name: String): Tensor[Float] = synchronized { owner(name).readFloat(name) }
  def readDouble(name: String): Tensor[Double] = synchronized { owner(name).readDouble(name) }
  def readLong(name: String): Tensor[Long] = synchronized { owner(name).readLong(name) }
  def readBoolean(name: String): Tensor[Boolean] = synchronized { owner(name).readBoolean(name) }
  override def close(): Unit = synchronized {
    if !closed then
      closed = true
      SafeTensorSet.closeAll(archives)
  }

object SafeTensorSet:
  private def closeAll(archives: Iterable[SafeTensors]): Unit =
    var failure: Option[Throwable] = None
    archives.foreach { archive =>
      try archive.close()
      catch case NonFatal(e) => failure match
        case None => failure = Some(e)
        case Some(first) => first.addSuppressed(e)
    }
    failure.foreach(throw _)

  def open(directory: Path, readLimits: ReadLimits = ReadLimits(), shardLimits: ShardLimits = ShardLimits()): SafeTensorSet =
    val single = directory.resolve("model.safetensors")
    val indexed = directory.resolve("model.safetensors.index.json")
    val hasSingle = Files.exists(single,LinkOption.NOFOLLOW_LINKS)
    val hasIndex = Files.exists(indexed,LinkOption.NOFOLLOW_LINKS)
    require(hasSingle != hasIndex, "exactly one single archive or safetensors index required")
    val index = if hasIndex then Some(SafeTensorsIndex.read(indexed,shardLimits,readLimits.maxTensors)) else None
    val names = index.map(_.weights.values.toVector.distinct.sorted).getOrElse(Vector("model.safetensors"))
    val acquired = ArrayBuffer.empty[SafeTensors]
    try
      val owners = Map.newBuilder[String,SafeTensors]
      val entries = Map.newBuilder[String,TensorInfo]
      var count = 0L
      var totalBytes = BigInt(0)
      names.foreach { filename =>
        val path = directory.resolve(filename)
        require(Files.isRegularFile(path), s"missing shard: $filename")
        val archive = SafeTensors.open(path,readLimits)
        acquired += archive
        index.foreach { ix =>
          val expected = ix.weights.iterator.collect { case (name,file) if file == filename => name }.toSet
          require(archive.tensors.keySet == expected, s"shard tensor assignments mismatch: $filename")
        }
        count += archive.tensors.size
        require(count <= readLimits.maxTensors, "aggregate tensor count exceeds limit")
        archive.tensors.foreach { (name,info) =>
          owners += name -> archive
          entries += name -> info
          totalBytes += info.end-info.begin
        }
      }
      index.flatMap(_.totalBytes).foreach(n => require(totalBytes == BigInt(n), "index total_size differs from payload bytes"))
      new SafeTensorSet(acquired.toVector,owners.result(),entries.result())
    catch case NonFatal(e) =>
      try closeAll(acquired)
      catch case NonFatal(closeFailure) => e.addSuppressed(closeFailure)
      throw e
