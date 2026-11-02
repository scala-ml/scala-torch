package scalatorch.nn

import scalatorch.*
import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, DataOutputStream, IOException}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path, StandardCopyOption}
import java.security.{DigestInputStream, DigestOutputStream, MessageDigest}
import scala.collection.mutable
import scala.util.Using

/** Versioned training-only format. Not a public model loader; no executable deserialization. */
object Checkpoint:
  private val magic = "STCKPT01".getBytes(UTF_8)
  private val maxTextBytes = 65536

  private def groups[A: Module](model: A): Vector[(Vector[String], Parameter)] =
    Module.namedParameters(model).groupMap(_._2)(_._1).toVector.map((p, names) => names.sorted -> p).sortBy(_._1.head)

  private def writeText(out: DataOutputStream, value: String): Unit =
    val bytes = value.getBytes(UTF_8)
    require(bytes.length <= maxTextBytes, "checkpoint text too long")
    out.writeInt(bytes.length)
    out.write(bytes)

  private def readText(in: DataInputStream): String =
    val n = in.readInt()
    require(n >= 0 && n <= maxTextBytes, "invalid checkpoint text length")
    val bytes = in.readNBytes(n)
    require(bytes.length == n, "truncated checkpoint text")
    UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(bytes)).toString

  private def membership(optimizer: Optimizer, grouped: Vector[(Vector[String], Parameter)]): Unit =
    val modelParameters = grouped.map(_._2).toSet
    require(optimizer.optimized.forall(modelParameters), "optimizer contains a parameter outside this model")

  def save[A: Module](model: A, optimizer: Optimizer, path: Path, step: Long): Unit =
    require(step >= 0, "negative training step")
    val grouped = groups(model)
    membership(optimizer, grouped)
    val slots = optimizer.snapshot
    optimizer.validate(slots)
    val byParameter = optimizer.optimized.zip(slots).toMap
    val destination = path.toAbsolutePath.normalize()
    val parent = destination.getParent
    Files.createDirectories(parent)
    val temporary = Files.createTempFile(parent, ".scala-torch-", ".tmp")
    try
      val hash = MessageDigest.getInstance("SHA-256")
      Using.resource(new DigestOutputStream(new BufferedOutputStream(Files.newOutputStream(temporary)), hash)) { stream =>
        val out = new DataOutputStream(stream)
        out.write(magic)
        out.writeLong(step)
        writeText(out, optimizer.kind)
        out.writeInt(optimizer.settings.size)
        optimizer.settings.foreach { (key, value) => writeText(out, key); out.writeDouble(value) }
        out.writeInt(grouped.size)
        grouped.foreach { (aliases, parameter) =>
          out.writeInt(aliases.size)
          aliases.foreach(writeText(out, _))
          out.writeBoolean(parameter.trainable)
          out.writeInt(parameter.data.shape.rank)
          parameter.data.shape.dims.foreach(out.writeLong)
          parameter.data.toFloatArray.foreach(v => out.writeInt(java.lang.Float.floatToRawIntBits(v)))
          val selected = byParameter.contains(parameter)
          out.writeBoolean(selected)
          if selected then
            val slot = byParameter(parameter)
            out.writeBoolean(slot.nonEmpty)
            slot.foreach { s =>
              out.writeLong(s.step)
              s.first.foreach(out.writeDouble)
              s.second.foreach(out.writeDouble)
            }
        }
        out.flush()
        val checksum = hash.digest()
        stream.on(false)
        out.write(checksum)
        out.flush()
      }
      // No non-atomic fallback: on unsupported filesystems leave the old destination intact.
      Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    finally Files.deleteIfExists(temporary)

  /** Bounded header discovery; callers must subsequently validate the complete payload. */
  private[nn] def adamWConfig(path: Path): AdamWConfig =
    try Using.resource(new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) { in =>
      require(in.readNBytes(magic.length).sameElements(magic),"unsupported checkpoint format")
      require(in.readLong()>=0,"invalid saved training step")
      require(readText(in)=="adamw","checkpoint is not AdamW")
      val names=Vector("learningRate","beta1","beta2","epsilon","weightDecay")
      require(in.readInt()==names.size,"optimizer setting count mismatch")
      val values=names.map { name =>
        require(readText(in)==name,s"optimizer setting mismatch: $name")
        in.readDouble()
      }
      AdamWConfig(values(0),values(1),values(2),values(3),values(4))
    } catch case error: IOException => throw IllegalArgumentException("invalid or unreadable checkpoint",error)

  /** Restores at an optimizer boundary, clears gradients, returns caller-supplied training step. */
  def load[A: Module](model: A, optimizer: Optimizer, path: Path): Long = loadInternal(model,optimizer,path,true)
  private[nn] def loadAdamWWeights[A: Module](model: A,optimizer: AdamW,path: Path): Long =
    loadInternal(model,optimizer,path,false)
  private def loadInternal[A: Module](model: A, optimizer: Optimizer, path: Path,retainMoments: Boolean): Long =
    val grouped = groups(model)
    membership(optimizer, grouped)
    val selected = optimizer.optimized.toSet
    val hash = MessageDigest.getInstance("SHA-256")
    val staged = try
      Using.resource(new DigestInputStream(new BufferedInputStream(Files.newInputStream(path)), hash)) { stream =>
        val in = new DataInputStream(stream)
        require(in.readNBytes(magic.length).sameElements(magic), "unsupported checkpoint format")
        val step = in.readLong()
        require(step >= 0, "invalid saved training step")
        require(readText(in) == optimizer.kind, "optimizer kind mismatch")
        val config = optimizer.settings
        require(in.readInt() == config.size, "optimizer setting count mismatch")
        config.foreach { (key, value) =>
          require(readText(in) == key && in.readDouble() == value, s"optimizer setting mismatch: $key")
        }
        require(in.readInt() == grouped.size, "model parameter count mismatch")
        val weights = mutable.Map.empty[String, Tensor[Float]]
        val slots = mutable.Map.empty[Parameter, Option[OptimizerSlot]]
        grouped.foreach { (aliases, parameter) =>
          require(in.readInt() == aliases.size, "parameter alias count mismatch")
          aliases.foreach(name => require(readText(in) == name, "parameter alias mismatch"))
          require(in.readBoolean() == parameter.trainable, "parameter trainability mismatch")
          val shape = parameter.data.shape
          require(in.readInt() == shape.rank, "parameter rank mismatch")
          shape.dims.foreach(dim => require(in.readLong() == dim, "parameter shape mismatch"))
          // Target shape, not unchecked file metadata, bounds these allocations.
          val values = Array.fill(shape.numel.toInt)(java.lang.Float.intBitsToFloat(in.readInt()))
          val tensor = Tensor.fromFloats(values, shape)
          aliases.foreach(name => weights(name) = tensor)
          require(in.readBoolean() == selected(parameter), "optimizer parameter membership mismatch")
          if selected(parameter) then
            val slot = if in.readBoolean() then
              val counter = in.readLong()
              require(counter > 0, "invalid optimizer counter")
              def readMoments(second: Boolean): Vector[Double] =
                def value(): Double =
                  val v=in.readDouble()
                  require(v.isFinite && (!second || v>=0),"invalid optimizer moments")
                  v
                if retainMoments then Vector.fill(values.length)(value())
                else
                  var i=0
                  while i<values.length do
                    value();i+=1
                  Vector.empty
              val first = readMoments(false)
              val second = readMoments(true)
              Some(OptimizerSlot(counter, first, second))
            else None
            slots(parameter) = slot
        }
        val saved = optimizer.optimized.map(slots)
        if retainMoments then optimizer.validate(saved)
        else
          require(saved.forall(_.isEmpty) || saved.forall(_.nonEmpty),"Adam checkpoint moment membership mismatch")
          require(saved.flatMap(_.map(_.step)).distinct.size<=1,"Adam checkpoint counters differ")
        val actualChecksum = hash.digest()
        stream.on(false)
        val expectedChecksum = in.readNBytes(32)
        require(expectedChecksum.length == 32 && MessageDigest.isEqual(actualChecksum, expectedChecksum), "checkpoint checksum mismatch")
        require(in.read() == -1, "checkpoint has trailing bytes")
        (step, weights.toMap, saved)
      }
    catch case e: IOException => throw IllegalArgumentException("invalid or unreadable checkpoint", e)
    val (step, weights, slots) = staged
    // All format, model and optimizer validation has completed before publishing any state.
    Module.loadStateDict(model, weights)
    if retainMoments then optimizer.restore(slots)
    step
