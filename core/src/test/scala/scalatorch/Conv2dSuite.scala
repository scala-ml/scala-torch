package scalatorch

class Conv2dSuite extends munit.FunSuite:
  private def t(values: Seq[Float],dims: Long*) = Tensor.fromFloats(values.toArray,Shape.from(dims))
  test("cross correlation preserves kernel orientation and handles strides and dilation") {
    val x=t((1 to 16).map(_.toFloat),1,1,4,4)
    val w=t(Seq(1f,2f,3f,4f),1,1,2,2)
    assertEquals(Conv2dOps.forward(x,w).toFloatArray.toVector,Vector(44f,54f,64f,84f,94f,104f,124f,134f,144f))
    assertEquals(Conv2dOps.forward(x,w,Conv2dConfig(stride=(2,2))).toFloatArray.toVector,Vector(44f,64f,124f,144f))
    assertEquals(Conv2dOps.forward(x,w,Conv2dConfig(dilation=(2,2))).toFloatArray.toVector,Vector(78f,88f,118f,128f))
  }
  test("grouped padded convolution isolates channels and accepts noncontiguous inputs") {
    val x=t(Seq(1f,2f,3f,4f,10f,20f,30f,40f),1,2,2,2).transpose(2,3)
    val w=t(Seq(2f,3f),2,1,1,1)
    val y=Conv2dOps.forward(x,w,Conv2dConfig(padding=(1,0),groups=2))
    assertEquals(y.shape,Shape.from(Seq(1,2,4,2)))
    assertEquals(y.toFloatArray.toVector,Vector(0f,0f,2f,6f,4f,8f,0f,0f,0f,0f,30f,90f,60f,120f,0f,0f))
  }
  test("empty batches and malformed geometry are checked before allocation") {
    val w=t(Seq(1f),1,1,1,1)
    assertEquals(Conv2dOps.forward(t(Seq.empty,0,1,2,2),w).shape,Shape.from(Seq(0,1,2,2)))
    intercept[IllegalArgumentException](Conv2dConfig(stride=(0,1)))
    intercept[IllegalArgumentException](Conv2dConfig(padding=(-1,0)))
    intercept[IllegalArgumentException](Conv2dConfig(dilation=(1,0)))
    intercept[IllegalArgumentException](Conv2dConfig(groups=0))
    intercept[IllegalArgumentException](Conv2dOps.forward(t(Seq(1f),1,1,1,1),t(Seq.fill(4)(1f),1,1,2,2)))
    intercept[IllegalArgumentException](Conv2dOps.forward(t(Seq(1f),1,1,1,1),w,Conv2dConfig(groups=2)))
    intercept[IllegalArgumentException](Conv2dOps.forward(t(Seq.empty,0,1,Long.MaxValue,1),w))
  }
