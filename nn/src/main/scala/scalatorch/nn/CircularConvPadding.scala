package scalatorch.nn

import scalatorch.*

final case class CircularConvConfig(
    tileX: Boolean = true,
    tileY: Boolean = true,
    startStep: Int = 0,
    stopStep: Int = -1,
    @volatile var active: Boolean = true
):
  def updateForStep(stepIndex0: Int, totalSteps: Int): Unit =
    if startStep > 0 || stopStep >= 0 then
      active = stepIndex0 >= startStep && (stopStep < 0 || stepIndex0 < stopStep)
    else
      active = tileX || tileY

  def enableForVaeDecode(): Unit =
    active = tileX || tileY

object CircularConvPadding:
  private val local = new ThreadLocal[Option[CircularConvConfig]]:
    override def initialValue(): Option[CircularConvConfig] = None

  def current: Option[CircularConvConfig] =
    local.get().filter(c => c.active && (c.tileX || c.tileY))

  def withConfig[A](config: Option[CircularConvConfig])(body: => A): A =
    val previous = local.get()
    local.set(config.filter(c => c.tileX || c.tileY))
    try body
    finally local.set(previous)

  def pad(input: Tensor[Float], padY: Int, padX: Int, tileX: Boolean, tileY: Boolean): Tensor[Float] =
    require(input.shape.rank == 4 && padY >= 0 && padX >= 0, "CircularConvPadding requires rank-4 NCHW input and non-negative padding")
    if padY == 0 && padX == 0 then input
    else
      val dims = input.shape.dims
      val n = dims(0).toInt
      val c = dims(1).toInt
      val h = dims(2).toInt
      val w = dims(3).toInt
      val outH = h + 2 * padY
      val outW = w + 2 * padX
      val src = input.toFloatArray
      val dst = new Array[Float](n * c * outH * outW)
      var plane = 0
      while plane < n * c do
        val srcBase = plane * h * w
        val dstBase = plane * outH * outW
        var oy = 0
        while oy < outH do
          val sy = oy - padY
          val validY = sy >= 0 && sy < h
          if validY || tileY then
            val wrappedY = if validY then sy else ((sy % h) + h) % h
            val srcRow = srcBase + wrappedY * w
            val dstRow = dstBase + oy * outW
            var ox = 0
            while ox < outW do
              val sx = ox - padX
              val validX = sx >= 0 && sx < w
              if validX || tileX then
                val wrappedX = if validX then sx else ((sx % w) + w) % w
                dst(dstRow + ox) = src(srcRow + wrappedX)
              ox += 1
          oy += 1
        plane += 1
      Tensor.fromFloats(dst, Shape.from(Seq(n.toLong, c.toLong, outH.toLong, outW.toLong)))
