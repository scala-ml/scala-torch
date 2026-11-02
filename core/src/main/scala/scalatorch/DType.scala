package scalatorch

enum DType[A]:
  case Float32 extends DType[Float]
  case Float64 extends DType[Double]
  case Int64 extends DType[Long]
  case Bool extends DType[Boolean]
