package scalatorch

enum Device:
  case Cpu
  case Cuda(index: Int)
