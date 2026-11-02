package scalatorch.models

import scalatorch.*
import scalatorch.nn.*
import java.io.{BufferedInputStream,BufferedOutputStream,DataInputStream,DataOutputStream,IOException,OutputStream}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path,StandardCopyOption}
import java.security.{DigestOutputStream,MessageDigest}
import scala.util.Using

/** Full resident AdamW checkpoint: architecture binding around the existing checked state format. */
private[models] object BackendLlamaCheckpoint:
  private val magic="STBADM01".getBytes(UTF_8)
  private val headerBytes=48L
  private def binding(model: LlamaModel): Array[Byte] =
    val hash=MessageDigest.getInstance("SHA-256")
    Using.resource(new DataOutputStream(new DigestOutputStream(OutputStream.nullOutputStream(),hash))) { out =>
      val c=model.config
      Seq(c.vocabSize,c.hiddenSize,c.intermediateSize,c.numLayers,c.numAttentionHeads,c.numKeyValueHeads,c.maxPositions).foreach(out.writeInt)
      out.writeDouble(c.rmsNormEps);out.writeDouble(c.ropeTheta);out.writeBoolean(c.tieWordEmbeddings)
      out.writeBoolean(c.bosTokenId.nonEmpty);c.bosTokenId.foreach(out.writeLong)
      out.writeInt(c.eosTokenIds.size);c.eosTokenIds.foreach(out.writeLong);out.writeDouble(c.initializerRange)
      out.writeInt(model.layers.size)
      def projection(layer: Linear): Unit =
        out.writeBoolean(layer.bias.nonEmpty);out.writeBoolean(layer.lora.nonEmpty)
        layer.lora.foreach(a => out.writeDouble(a.alpha))
      model.layers.foreach { layer =>
        out.writeDouble(layer.inputNorm.epsilon);out.writeDouble(layer.postNorm.epsilon)
        Seq(layer.attention.qProj,layer.attention.kProj,layer.attention.vProj,layer.attention.oProj,
          layer.mlp.gateProj,layer.mlp.upProj,layer.mlp.downProj).foreach(projection)
      }
      out.writeDouble(model.norm.epsilon);projection(model.lmHead)
    }
    hash.digest()
  def save(model: LlamaModel,weights: Map[String,Tensor[Float]],moments: Map[String,AdamWMoments],
      config: AdamWConfig,path: Path,step: Long,maxBytes: Long): Unit =
    require(maxBytes>=headerBytes,"invalid checkpoint byte budget")
    val destination=path.toAbsolutePath.normalize()
    Files.createDirectories(destination.getParent)
    val state=Files.createTempFile(destination.getParent,".backend-adam-state-",".tmp")
    try
      BackendAdamCheckpoint.save(model,weights,moments,config,state,step)
      val length=Files.size(state)
      require(length<=maxBytes-headerBytes,"checkpoint byte budget exceeded")
      val temporary=Files.createTempFile(destination.getParent,".backend-adam-bundle-",".tmp")
      try
        Using.resource(new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(temporary)))) { out =>
          out.write(magic);out.write(binding(model));out.writeLong(length);Files.copy(state,out)
        }
        Files.move(temporary,destination,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
      finally Files.deleteIfExists(temporary)
    finally Files.deleteIfExists(state)
  def load(model: LlamaModel,config: AdamWConfig,path: Path,maxBytes: Long): BackendAdamCheckpoint.Saved =
    require(maxBytes>=headerBytes,"invalid checkpoint byte budget")
    val state=Files.createTempFile("backend-adam-load-",".tmp")
    try
      try Using.resource(new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) { in =>
        require(in.readNBytes(8).sameElements(magic),"unsupported backend checkpoint format")
        require(MessageDigest.isEqual(in.readNBytes(32),binding(model)),"checkpoint model configuration mismatch")
        val length=in.readLong()
        require(length>=0 && length<=maxBytes-headerBytes,"checkpoint byte budget exceeded")
        Using.resource(new BufferedOutputStream(Files.newOutputStream(state))) { out =>
          val buffer=new Array[Byte](65536)
          var remaining=length
          while remaining>0 do
            val n=in.read(buffer,0,math.min(remaining,buffer.length.toLong).toInt)
            require(n>0,"truncated backend checkpoint")
            out.write(buffer,0,n);remaining-=n
        }
        require(in.read()== -1,"backend checkpoint has trailing bytes")
      }
      catch case error: IOException => throw IllegalArgumentException("invalid or unreadable backend checkpoint",error)
      BackendAdamCheckpoint.load(model,config,state)
    finally Files.deleteIfExists(state)
