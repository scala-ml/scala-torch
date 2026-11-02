package scalatorch

/** Explicit execution and ownership; operations do not record gradients or transfer implicitly.
  * Arithmetic consumes no inputs. Results are independently releasable.
  * Add/multiply require equal shapes; RMSNorm normalizes the final axis.
  */
trait FloatBackend[T]:
  def upload(source: Tensor[Float]): T
  def download(value: T): Tensor[Float]
  def shape(value: T): Shape
  def release(value: T): Unit
  def reshape(value: T, shape: Shape): T
  def transpose(value: T, dim0: Int, dim1: Int): T
  /** IDs are explicitly host-side Int64 tensors; output appends the table width. */
  def embedding(table: T, indices: Tensor[Long]): T
  def matmul(a: T, b: T): T
  def add(a: T, b: T): T
  def multiply(a: T, b: T): T
  /** Elementwise Float32 scalar multiplication, including IEEE nonfinite values. */
  def scale(value: T, scalar: Float): T
  def fill(shape: Shape, value: Float): T
  /** Sum all elements into a rank-zero Float32 output, accumulating in Double. */
  def sum(value: T): T
  /** Repeat a rank-zero backend value without a host transfer. */
  def expandScalar(value: T, shape: Shape): T
  def silu(value: T): T
  def rmsNorm(value: T, weight: T, epsilon: Double = 1e-6): T
  def rope(value: T, positionStart: Int = 0, theta: Double = 10000.0): T
  def causalAttention(query: T,key: T,value: T,queryStart: Int = 0,maxScoreElements: Long = 16777216L): T
  def attentionCache(batchSize: Int,heads: Int,capacity: Int,keyWidth: Int,valueWidth: Int,
      maxBytes: Long = 268435456L): AttentionCache[T]
