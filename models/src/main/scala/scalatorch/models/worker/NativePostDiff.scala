package scalatorch.models.worker

import scalatorch.models.PostDiffConfig
import NativeJson.*

/** Experimental worker-only field; not a public engine feature or a supported replay format. */
private[worker] object NativePostDiff:
  def parse(value: Option[NativeJson]): Option[PostDiffConfig] = value match
    case None | Some(Null) => None
    case Some(Str(encoded)) =>
      require(encoded.length<=256,"PostDiff experimental options are too long")
      val decoded=NativeJson.parse(encoded)
      require(decoded.isInstanceOf[Obj],"PostDiff experimental options must be an object")
      parse(Some(decoded))
    case Some(Obj(fields)) =>
      require(fields.keySet==Set("scale","portion"),"PostDiff requires scale and portion")
      def number(key: String): Double = fields(key) match
        case Number(v) => v.toDouble
        case _ => throw new IllegalArgumentException(s"PostDiff $key must be numeric")
      val scale=number("scale")
      val portion=number("portion")
      require(scale.isFinite && scale>0 && scale<1 && portion.isFinite && portion>0 && portion<1,
        "PostDiff scale and portion must be strictly between 0 and 1")
      Some(PostDiffConfig(scale,portion))
    case _ => throw new IllegalArgumentException("invalid PostDiff experimental options")
