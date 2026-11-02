package scalatorch

/** Explicit inference execution and ownership; no autograd or implicit device transfer.
  * Arithmetic consumes no inputs. Results are independently releasable.
  * Add/multiply require equal shapes; RMSNorm normalizes the final axis.
  */
trait FloatBackend[T]:
  def upload(source: Tensor[Float]): T
  def download(value: T): Tensor[Float]
  def shape(value: T): Shape
  def release(value: T): Unit
  def matmul(a: T, b: T): T
  def add(a: T, b: T): T
  def multiply(a: T, b: T): T
  def silu(value: T): T
  def rmsNorm(value: T, weight: T, epsilon: Double = 1e-6): T
