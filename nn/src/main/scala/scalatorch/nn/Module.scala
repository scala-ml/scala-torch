package scalatorch.nn

import scalatorch.*
import scala.deriving.Mirror
import scala.compiletime.{constValueTuple, erasedValue, summonInline}

/** Derive for a model case class; fields must have explicit parameter-tree instances. */
trait Module[A]:
  def namedParameters(model: A): Vector[(String, Parameter)]

object Module:
  given Module[Conv2dConfig] with
    def namedParameters(model: Conv2dConfig) = Vector.empty
  given Module[Long] with
    def namedParameters(model: Long) = Vector.empty
  given Module[Int] with
    def namedParameters(model: Int) = Vector.empty
  given Module[Double] with
    def namedParameters(model: Double) = Vector.empty
  given Module[Parameter] with
    def namedParameters(model: Parameter) = Vector("" -> model)
  given [A](using inner: Module[A]): Module[Option[A]] with
    def namedParameters(model: Option[A]) = model.toVector.flatMap(inner.namedParameters)
  given [A](using inner: Module[A]): Module[Vector[A]] with
    def namedParameters(model: Vector[A]) = model.zipWithIndex.flatMap { (value, index) =>
      inner.namedParameters(value).map((name, p) => join(index.toString, name) -> p)
    }

  private def join(prefix: String, name: String): String = if name.isEmpty then prefix else s"$prefix.$name"
  private inline def instances[T <: Tuple]: List[Module[?]] = inline erasedValue[T] match
    case _: EmptyTuple => Nil
    case _: (head *: tail) => summonInline[Module[head]] :: instances[tail]

  inline def derived[A](using mirror: Mirror.ProductOf[A]): Module[A] =
    val names = constValueTuple[mirror.MirroredElemLabels].toList.asInstanceOf[List[String]]
    val fields = instances[mirror.MirroredElemTypes]
    product(names, fields)

  def product[A](names: List[String], fields: List[Module[?]]): Module[A] = new Module[A]:
    def namedParameters(model: A): Vector[(String, Parameter)] =
      val value = model.asInstanceOf[Product]
      names.zip(fields).zipWithIndex.toVector.flatMap { case ((prefix, field), index) =>
        field.asInstanceOf[Module[Any]].namedParameters(value.productElement(index)).map { (name, p) =>
          join(prefix, name) -> p
        }
      }

  def namedParameters[A](model: A)(using module: Module[A]): Vector[(String, Parameter)] =
    val result = module.namedParameters(model)
    require(result.map(_._1).distinct.size == result.size, "duplicate parameter names")
    result

  def parameters[A: Module](model: A): Vector[Parameter] = namedParameters(model).map(_._2).distinct
  def stateDict[A: Module](model: A): Map[String, Tensor[Float]] = namedParameters(model).map((name, p) => name -> p.data).toMap

  def loadStateDict[A: Module](model: A, state: Map[String, Tensor[Float]]): Unit =
    val named = namedParameters(model)
    require(state.keySet == named.map(_._1).toSet, "state dict keys differ from model")
    val updates = scala.collection.mutable.LinkedHashMap.empty[Parameter, Tensor[Float]]
    named.foreach { (name, parameter) =>
      val next = state(name)
      require(next.shape == parameter.data.shape && next.dtype == DType.Float32 && next.device == Device.Cpu,
        s"state tensor mismatch: $name")
      updates.get(parameter).foreach { previous =>
        val a = previous.toFloatArray.map(java.lang.Float.floatToRawIntBits)
        val b = next.toFloatArray.map(java.lang.Float.floatToRawIntBits)
        require(a.sameElements(b), s"conflicting tied parameter: $name")
      }
      updates(parameter) = next
    }
    updates.foreach((parameter, next) => parameter.load(next))
