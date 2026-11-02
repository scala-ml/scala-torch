package scalatorch.cpu

import scalatorch.*
import com.github.kmizu.numscala.cpu.{MatrixF32, ScalarF32Kernels, Transpose, Workspace}

/** Float32 GEMM adapter. Unsupported layouts are explicitly packed, never reinterpreted. */
object NumscalaBackend:
  private def matrix(x: Tensor[Float]): (MatrixF32, Transpose) =
    val rows = x.shape.dims(0).toInt
    val cols = x.shape.dims(1).toInt
    val strides = x.layout.strides
    val offset = x.layout.offset.toInt
    if strides(1) == 1 && strides(0) >= math.max(cols, 1) && strides(0) <= Int.MaxValue then
      (MatrixF32(x.storage.data, offset, rows, cols, strides(0).toInt), Transpose.No)
    else if strides(0) == 1 && strides(1) >= math.max(rows, 1) && strides(1) <= Int.MaxValue then
      (MatrixF32(x.storage.data, offset, cols, rows, strides(1).toInt), Transpose.Yes)
    else (MatrixF32.wrap(x.toFloatArray, rows, cols), Transpose.No)

  def matmul(a: Tensor[Float], b: Tensor[Float]): Tensor[Float] =
    val shape = Ops.matmulShape(a, b)
    // Empty outputs avoid converting potentially huge dimensions to Int.
    if shape.numel == 0 then return Tensor.fromFloats(Array.emptyFloatArray, shape)
    require((a.shape.dims ++ b.shape.dims).forall(_ <= Int.MaxValue), "GEMM dimension exceeds Int indexing")
    val (am, at) = matrix(a)
    val (bm, bt) = matrix(b)
    val out = MatrixF32.zeros(shape.dims(0).toInt, shape.dims(1).toInt)
    val workspace=new Workspace()
    workspace.reserveFloats(ScalarF32Kernels.gemmWorkspaceFloats(out.rows,out.cols,a.shape.dims(1).toInt,at,bt))
    ScalarF32Kernels.gemmInto(am, at, bm, bt, out, workspace)
    Tensor.fromFloats(out.data, shape)
