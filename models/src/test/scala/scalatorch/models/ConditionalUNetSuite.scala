package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Variable,Autograd}
import scalatorch.nn.*

class ConditionalUNetSuite extends munit.FunSuite:
  private def model(trainable: Boolean = true): ConditionalUNet2d =
    var seed=100L
    def next(): Long = { seed+=1;seed }
    def parameter(t: Tensor[Float])=Parameter(t,trainable)
    def linear(in: Int,out: Int,seed: Long): Linear =
      val l=Linear.init(in,out,seed)
      l.copy(weight=parameter(l.weight.data),bias=l.bias.map(p => parameter(p.data)))
    def group(width: Int,groups: Int,epsilon: Double): GroupNorm =
      val n=GroupNorm.init(width,groups,epsilon)
      n.copy(weight=n.weight.map(p => parameter(p.data)),bias=n.bias.map(p => parameter(p.data)))
    def norm(width: Int,epsilon: Double): LayerNorm =
      val n=LayerNorm.init(width,epsilon)
      n.copy(weight=n.weight.map(p => parameter(p.data)),bias=n.bias.map(p => parameter(p.data)))
    def conv(in: Int,out: Int,k: Int=3,stride: Int=1): Conv2d =
      val random=new java.util.Random(next())
      val shape=Shape.from(Seq(out.toLong,in.toLong,k.toLong,k.toLong))
      Conv2d(parameter(Tensor.fromFloats(Array.fill(shape.numel.toInt)((random.nextGaussian()*.08).toFloat),shape)),
        Some(parameter(Tensor.fromFloats(Array.fill(out)(.01f),Shape.from(Seq(out.toLong))))),
        Conv2dConfig(stride=(stride,stride),padding=(k/2,k/2)))
    def res(in: Int,out: Int)=DiffusionResnet2d(group(in,1,epsilon=.1),conv(in,out),
      Some(linear(4,out,next())),group(out,1,epsilon=.1),conv(out,out),
      if in==out then None else Some(conv(in,out,1)))
    def attention(width: Int): DiffusionSpatialTransformer =
      def att(context: Int)=DiffusionAttention(linear(width,width,next()),linear(context,width,next()),
        linear(context,width,next()),linear(width,width,next()),1)
      val block=DiffusionTransformerBlock(norm(width,epsilon=.1),att(width),norm(width,epsilon=.1),
        att(3),norm(width,epsilon=.1),DiffusionFeedForward(linear(width,2*width,next()),linear(width,width,next())))
      DiffusionSpatialTransformer(group(width,1,epsilon=.1),conv(width,width,1),Vector(block),conv(width,width,1))
    def layer(in: Int,out: Int,att: Boolean=false)=DiffusionUNetLayer(res(in,out),if att then Some(attention(out)) else None)
    ConditionalUNet2d(conv(1,2),DiffusionTimeEmbedding(linear(4,4,next()),linear(4,4,next())),
      Vector(DiffusionDownBlock(Vector(layer(2,2,true)),Some(DiffusionDownsample2d(conv(2,2,3,2)))),
        DiffusionDownBlock(Vector(layer(2,4)),None)),
      DiffusionMidBlock(res(4,4),attention(4),res(4,4)),
      Vector(DiffusionUpBlock(Vector(layer(8,4),layer(6,4)),Some(DiffusionUpsample2d(conv(4,4)))),
        DiffusionUpBlock(Vector(layer(6,2,true),layer(4,2)),None)),
      group(2,1,epsilon=.1),conv(2,1))
  private val xs=Shape.from(Seq(1,1,3,5))
  private val cs=Shape.from(Seq(1,2,3))
  private def times(values: Float*)=Tensor.fromFloats(values.toArray,Shape.from(Seq(values.size.toLong)))
  test("two-resolution conditioned U-Net restores odd image dimensions and connects all parameters") {
    val m=model()
    val xa=Array.tabulate(15)(i => math.sin(i*.4).toFloat)
    val ca=Array.tabulate(6)(i => math.cos(i*.3).toFloat)
    val x=Variable(Tensor.fromFloats(xa,xs),true);val c=Variable(Tensor.fromFloats(ca,cs),true)
    val y=m(x,times(.5f),c)
    assertEquals(y.data.shape,xs)
    assert(y.data.toFloatArray.forall(_.isFinite))
    val seed=Tensor.fromFloats(Array.tabulate(15)(i => math.cos(i*.6).toFloat),xs)
    y.backward(seed)
    val parameters=Module.parameters(m)
    assert(parameters.size>50)
    assert(parameters.forall(p => p.grad.exists(_.toFloatArray.forall(_.isFinite))))
    assert(c.grad.get.toFloatArray.exists(g => math.abs(g)>1e-7))
    val gradients=parameters.map(p => p -> p.grad.get.toFloatArray).toMap
    val originals=parameters.map(p => p -> p.data).toMap
    val dx=x.grad.get.toFloatArray;val dc=c.grad.get.toFloatArray
    for direction <- 0 until 3 do
      val random=new java.util.Random(direction+300)
      def vector(n: Int)=Array.fill(n)((random.nextDouble()*.2-.1).toFloat)
      val xd=vector(xa.length);val cd=vector(ca.length)
      val pd=parameters.map(p => p -> vector(p.data.shape.numel.toInt)).toMap
      val expected=dx.zip(xd).map((a,b) => a.toDouble*b).sum+dc.zip(cd).map((a,b) => a.toDouble*b).sum+
        parameters.map(p => gradients(p).zip(pd(p)).map((a,b) => a.toDouble*b).sum).sum
      def score(step: Float): Double =
        parameters.foreach(p => p.load(Tensor.fromFloats(originals(p).toFloatArray.zip(pd(p)).map((a,b) => a+step*b),p.data.shape)))
        val xx=Variable(Tensor.fromFloats(xa.zip(xd).map((a,b) => a+step*b),xs))
        val cc=Variable(Tensor.fromFloats(ca.zip(cd).map((a,b) => a+step*b),cs))
        Autograd.noGrad(m(xx,times(.5f),cc)).data.toFloatArray.zip(seed.toFloatArray).map((a,b) => a.toDouble*b).sum
      val plus=score(.002f);val minus=score(-.002f)
      parameters.foreach(p => p.load(originals(p)))
      assertEqualsDouble(expected,(plus-minus)/.004,.002)
    val changed=Autograd.noGrad(m(Variable(Tensor.fromFloats(xa,xs)),times(8f),Variable(Tensor.fromFloats(ca,cs)))).data
    assert(changed.toFloatArray.toVector!=y.data.toFloatArray.toVector)
  }
  test("scalar timestep expands over batches and invalid topology rejects at construction") {
    val m=model()
    val image=Variable(Tensor.fromFloats(Array.fill(30)(.2f),Shape.from(Seq(2,1,3,5))))
    val context=Variable(Tensor.fromFloats(Array.fill(12)(.3f),Shape.from(Seq(2,2,3))))
    val scalar=Autograd.noGrad(m(image,times(2f),context)).data
    val batched=Autograd.noGrad(m(image,times(2f,2f),context)).data
    assertEquals(scalar.toFloatArray.toVector,batched.toFloatArray.toVector)
    intercept[IllegalArgumentException](m(image,times(1f,2f,3f),context))
    intercept[IllegalArgumentException](m.copy(upBlocks=m.upBlocks.drop(1)))
    intercept[IllegalArgumentException](m.copy(upBlocks=m.upBlocks.updated(0,m.upBlocks.head.copy(layers=m.upBlocks.head.layers.drop(1)))))
    m.convOut.weight.load(Tensor.fromFloats(Array.fill(m.convOut.weight.data.shape.numel.toInt)(0f),m.convOut.weight.data.shape))
    m.convOut.bias.foreach(p => p.load(Tensor.fromFloats(Array.fill(p.data.shape.numel.toInt)(0f),p.data.shape)))
    assert(Autograd.noGrad(m(image,times(2f),context)).data.toFloatArray.forall(_==0f))
  }
  test("skip counts and channels cannot hide a resolution-level mismatch") {
    val m=model()
    val early=m.upBlocks.head.copy(layers=m.upBlocks.head.layers :+ m.upBlocks.head.layers.last)
    val late=m.upBlocks.last.copy(layers=m.upBlocks.last.layers.take(1))
    intercept[IllegalArgumentException](m.copy(upBlocks=Vector(early,late)))
  }
  test("frozen U-Net inference does not track constant timestep features") {
    val m=model(trainable=false)
    assert(Module.parameters(m).forall(!_.trainable))
    val x=Variable(Tensor.fromFloats(Array.fill(15)(.2f),xs),requiresGrad=false)
    val c=Variable(Tensor.fromFloats(Array.fill(6)(.3f),cs),requiresGrad=false)
    assert(!m(x,times(.5f),c).requiresGrad)
  }
