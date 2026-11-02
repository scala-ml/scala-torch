package scalatorch.models

import scalatorch.*
import scalatorch.io.SafeTensorSet
import scalatorch.nn.*
import java.nio.file.Path
import scala.collection.mutable
import scala.util.Using

/** CPU CLIP text conditioning from single or indexed Hugging Face safetensors. */
object PretrainedClip:
  def load(directory: Path,trainable: Boolean=false,maxParameters: Long=1000000000L,
      maxScoreElements: Long=16777216L,maxHiddenElements: Long=16777216L): ClipTextModel =
    require(maxParameters>=0 && maxScoreElements>=0 && maxHiddenElements>=0,"negative CLIP allocation limit")
    val c=ClipTextConfig.read(directory.resolve("config.json"))
    Using.resource(SafeTensorSet.open(directory)) { archive =>
      val total=archive.tensors.values.foldLeft(BigInt(0))((n,t) => n+t.shape.numel)
      require(total<=maxParameters,"CLIP weights exceed parameter budget")
      val used=mutable.Set.empty[String]
      val positionIds="text_model.embeddings.position_ids"
      archive.tensors.get(positionIds).foreach { info =>
        require(info.shape==Shape.from(Seq(1L,c.maxPositions.toLong)),"CLIP position_ids shape mismatch")
        val ids=archive.readLong(positionIds).toLongArray
        require(ids.indices.forall(i => ids(i)==i.toLong),"CLIP position_ids must be canonical consecutive positions")
        used+=positionIds
      }
      def parameter(name: String,dims: Int*): Parameter =
        val info=archive.tensors.getOrElse(name,throw IllegalArgumentException(s"missing CLIP weight: $name"))
        require(info.shape==Shape.from(dims.map(_.toLong)),s"CLIP weight shape mismatch: $name")
        require(used.add(name),s"duplicate CLIP weight use: $name")
        Parameter(archive.readFloat(name),trainable)
      def linear(name: String,in: Int,out: Int): Linear =
        Linear(parameter(s"$name.weight",out,in),Some(parameter(s"$name.bias",out)))
      def norm(name: String): LayerNorm =
        LayerNorm(c.width,Some(parameter(s"$name.weight",c.width)),Some(parameter(s"$name.bias",c.width)),c.epsilon)
      val tokens=Embedding(parameter("text_model.embeddings.token_embedding.weight",c.vocabulary,c.width))
      val positions=Embedding(parameter("text_model.embeddings.position_embedding.weight",c.maxPositions,c.width))
      val layers=Vector.tabulate(c.layers) { i =>
        val p=s"text_model.encoder.layers.$i"
        ClipTextLayer(norm(s"$p.layer_norm1"),linear(s"$p.self_attn.q_proj",c.width,c.width),
          linear(s"$p.self_attn.k_proj",c.width,c.width),linear(s"$p.self_attn.v_proj",c.width,c.width),
          linear(s"$p.self_attn.out_proj",c.width,c.width),norm(s"$p.layer_norm2"),
          linear(s"$p.mlp.fc1",c.width,c.intermediate),linear(s"$p.mlp.fc2",c.intermediate,c.width),c.heads,c.activation)
      }
      val projection=c.projectionWidth.map(width => Linear(parameter("text_projection.weight",width,c.width),None))
      val model=ClipTextModel(tokens,positions,layers,norm("text_model.final_layer_norm"),maxScoreElements,maxHiddenElements,
        projection,c.eosId.toLong)
      require(used.toSet==archive.tensors.keySet,s"unused CLIP weights: ${(archive.tensors.keySet--used).toVector.sorted.mkString(", ")}")
      model
    }
