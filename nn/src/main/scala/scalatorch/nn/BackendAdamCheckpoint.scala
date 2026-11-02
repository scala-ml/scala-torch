package scalatorch.nn

import scalatorch.*
import java.nio.file.Path

/** Bridge resident snapshots to the existing checked checkpoint format without mutating the source model. */
private[scalatorch] object BackendAdamCheckpoint:
  final case class Saved(step: Long,weights: Map[String,Tensor[Float]],moments: Map[String,AdamWMoments])
  private final case class Host(named: Vector[(String,Parameter)])
  private given Module[Host] with
    def namedParameters(value: Host) = value.named
  private def host[A: Module](model: A): Host =
    val named=Module.namedParameters(model)
    val copies=named.map(_._2).distinct.map(p => p -> Parameter(p.data,p.trainable)).toMap
    Host(named.map((name,p) => name -> copies(p)))
  private def optimizer(value: Host,c: AdamWConfig) =
    new AdamW(Module.parameters(value),c.learningRate,c.beta1,c.beta2,c.epsilon,c.weightDecay)
  private def bits(value: Tensor[Double]) = value.toDoubleArray.toVector.map(java.lang.Double.doubleToRawLongBits)
  def save[A: Module](model: A,weights: Map[String,Tensor[Float]],moments: Map[String,AdamWMoments],
      config: AdamWConfig,path: Path,step: Long): Unit =
    require(weights.values.forall(_.toFloatArray.forall(_.isFinite)),"nonfinite checkpoint weights")
    val value=host(model)
    Module.loadStateDict(value,weights)
    val opt=optimizer(value,config)
    val trainableNames=value.named.filter(_._2.trainable).map(_._1).toSet
    require(moments.isEmpty || moments.keySet==trainableNames,"Adam checkpoint moment membership mismatch")
    require(moments.values.map(_.step).toSet.size<=1,"Adam checkpoint counters differ")
    val aliases=value.named.groupMap(_._2)(_._1)
    val slots=opt.optimized.map { p =>
      val saved=aliases(p).flatMap(moments.get)
      saved.headOption.map { first =>
        saved.foreach { other =>
          other.validate()
          require(other.shape==p.data.shape && other.step==first.step && bits(other.first)==bits(first.first) &&
            bits(other.second)==bits(first.second),"Adam checkpoint tied moments differ")
        }
        OptimizerSlot(first.step,first.first.toDoubleArray.toVector,first.second.toDoubleArray.toVector)
      }
    }
    opt.restore(slots)
    Checkpoint.save(value,opt,path,step)
  def load[A: Module](model: A,config: AdamWConfig,path: Path): Saved =
    val value=host(model)
    val opt=optimizer(value,config)
    val step=Checkpoint.load(value,opt,path)
    val slots=opt.optimized.zip(opt.snapshot).toMap
    val saved=value.named.flatMap { (name,p) => slots.get(p).flatten.map { s =>
      name -> AdamWMoments(s.step,Tensor.fromDoubles(s.first.toArray,p.data.shape),Tensor.fromDoubles(s.second.toArray,p.data.shape))
    }}.toMap
    val expected=value.named.filter(_._2.trainable).map(_._1).toSet
    require(saved.isEmpty || saved.keySet==expected,"Adam checkpoint moment membership mismatch")
    require(saved.values.map(_.step).toSet.size<=1,"Adam checkpoint counters differ")
    Saved(step,Module.stateDict(value),saved)
