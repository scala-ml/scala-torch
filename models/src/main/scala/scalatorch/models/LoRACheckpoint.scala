package scalatorch.models

import scalatorch.nn.*
import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, IOException, OutputStream}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path, StandardCopyOption}
import java.security.{DigestOutputStream, MessageDigest}
import scala.util.Using

/** Adapter-only training checkpoint, bound to the exact frozen base and configuration.
  * Call at an optimizer boundary; concurrent model mutation is unsupported.
  */
object LoRACheckpoint:
  private val magic = "STLORA01".getBytes(UTF_8)
  private val headerBytes = 48L
  private val defaultMaxBytes = 1073741824L

  private def fingerprint(value: LlamaLoRA): Array[Byte] =
    val hash = MessageDigest.getInstance("SHA-256")
    Using.resource(new DataOutputStream(new BufferedOutputStream(
        new DigestOutputStream(OutputStream.nullOutputStream(),hash)))) { out =>
      def text(s: String): Unit =
        val bytes=s.getBytes(UTF_8)
        out.writeInt(bytes.length);out.write(bytes)
      text("scalatorch-llama-lora-base-v1")
      val c=value.model.config
      Seq(c.vocabSize,c.hiddenSize,c.intermediateSize,c.numLayers,c.numAttentionHeads,c.numKeyValueHeads,c.maxPositions).foreach(out.writeInt)
      out.writeDouble(c.rmsNormEps);out.writeDouble(c.ropeTheta);out.writeBoolean(c.tieWordEmbeddings)
      out.writeBoolean(c.bosTokenId.nonEmpty);c.bosTokenId.foreach(out.writeLong)
      out.writeInt(c.eosTokenIds.size);c.eosTokenIds.foreach(out.writeLong)
      out.writeDouble(c.initializerRange)
      // Actual modules may have been constructed independently of the config.
      out.writeInt(value.model.layers.size)
      value.model.layers.foreach { layer =>
        out.writeDouble(layer.inputNorm.epsilon);out.writeDouble(layer.postNorm.epsilon)
      }
      out.writeDouble(value.model.norm.epsilon)
      val adapters=Module.parameters(value).toSet
      val base=Module.namedParameters(value.model).filterNot((_,p) => adapters(p))
      val groups=base.groupMap(_._2)(_._1).toVector.map((p,names) => names.sorted -> p).sortBy(_._1.head)
      out.writeInt(groups.size)
      groups.foreach { (names,p) =>
        require(!p.trainable,"LoRA checkpoint requires a frozen base")
        out.writeInt(names.size);names.foreach(text)
        out.writeInt(p.data.shape.rank);p.data.shape.dims.foreach(out.writeLong)
        p.data.toFloatArray.foreach(v => out.writeInt(java.lang.Float.floatToRawIntBits(v)))
      }
      out.writeInt(value.adapters.size)
      value.adapters.foreach { (name,a) =>
        text(name);out.writeDouble(a.alpha)
        a.a.data.shape.dims.foreach(out.writeLong);a.b.data.shape.dims.foreach(out.writeLong)
      }
    }
    hash.digest()

  def save(value: LlamaLoRA, optimizer: Optimizer, path: Path, step: Long,
      maxBytes: Long = defaultMaxBytes): Unit =
    require(maxBytes >= headerBytes,"invalid LoRA checkpoint byte budget")
    val binding=fingerprint(value)
    val destination=path.toAbsolutePath.normalize()
    Files.createDirectories(destination.getParent)
    val state=Files.createTempFile(destination.getParent,".lora-state-",".tmp")
    try
      Checkpoint.save(value,optimizer,state,step)
      val length=Files.size(state)
      require(length <= maxBytes-headerBytes,"LoRA checkpoint byte budget exceeded")
      val temporary=Files.createTempFile(destination.getParent,".lora-bundle-",".tmp")
      try
        Using.resource(new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(temporary)))) { out =>
          out.write(magic);out.write(binding);out.writeLong(length)
          Files.copy(state,out)
        }
        Files.move(temporary,destination,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
      finally Files.deleteIfExists(temporary)
    finally Files.deleteIfExists(state)

  /** All envelope and inner checkpoint checks complete before weights or optimizer change. */
  def load(value: LlamaLoRA, optimizer: Optimizer, path: Path,
      maxBytes: Long = defaultMaxBytes): Long =
    require(maxBytes >= headerBytes,"invalid LoRA checkpoint byte budget")
    val binding=fingerprint(value)
    val state=Files.createTempFile("scalatorch-lora-load-",".tmp")
    try
      try
        Using.resource(new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) { in =>
          require(in.readNBytes(magic.length).sameElements(magic),"unsupported LoRA checkpoint format")
          require(MessageDigest.isEqual(in.readNBytes(32),binding),"LoRA base/configuration mismatch")
          val length=in.readLong()
          require(length >= 0 && length <= maxBytes-headerBytes,"LoRA checkpoint byte budget exceeded")
          Using.resource(new BufferedOutputStream(Files.newOutputStream(state))) { out =>
            val buffer=new Array[Byte](65536)
            var remaining=length
            while remaining > 0 do
              val n=in.read(buffer,0,math.min(remaining,buffer.length.toLong).toInt)
              require(n > 0,"truncated LoRA checkpoint")
              out.write(buffer,0,n);remaining-=n
          }
          require(in.read() == -1,"LoRA checkpoint has trailing bytes")
        }
        Checkpoint.load(value,optimizer,state)
      catch case e: IOException => throw IllegalArgumentException("invalid or unreadable LoRA checkpoint",e)
    finally Files.deleteIfExists(state)
