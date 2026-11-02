package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.nn.*

class VaeGraphSuite extends munit.FunSuite:
  private def pair(trainable: Boolean=true): (VaeEncoder,VaeDecoder) =
    var seed=900L
    def p(t: Tensor[Float])=Parameter(t,trainable)
    def conv(in: Int,out: Int,k: Int=3,stride: Int=1,padding: Int=1): Conv2d =
      seed+=1;val random=new java.util.Random(seed)
      val shape=Shape.from(Seq(out,in,k,k))
      Conv2d(p(Tensor.fromFloats(Array.fill(shape.numel.toInt)((random.nextGaussian()*.08).toFloat),shape)),
        Some(p(Tensor.fromFloats(Array.fill(out)(.01f),Shape.from(Seq(out))))),Conv2dConfig(stride=(stride,stride),padding=(padding,padding)))
    def norm(n: Int): GroupNorm =
      val g=GroupNorm.init(n,1,.1)
      g.copy(weight=g.weight.map(x => p(x.data)),bias=g.bias.map(x => p(x.data)))
    def linear(n: Int): Linear =
      seed+=1;val l=Linear.init(n,n,seed)
      l.copy(weight=p(l.weight.data),bias=l.bias.map(x => p(x.data)))
    def res(in: Int,out: Int)=DiffusionResnet2d(norm(in),conv(in,out),None,norm(out),conv(out,out),
      if in==out then None else Some(conv(in,out,1,padding=0)))
    def mid(n: Int)=VaeMidBlock(res(n,n),Some(VaeAttention2d(norm(n),DiffusionAttention(linear(n),linear(n),linear(n),linear(n),1))),res(n,n))
    val encoder=VaeEncoder(conv(3,2),Vector(VaeDownBlock(Vector(res(2,2)),Some(DiffusionDownsample2d(conv(2,2,stride=2,padding=0)))),
      VaeDownBlock(Vector(res(2,4)),None)),mid(4),norm(4),conv(4,2),Some(conv(2,2,1,padding=0)))
    val decoder=VaeDecoder(conv(1,4),mid(4),Vector(VaeUpBlock(Vector(res(4,4),res(4,4)),Some(DiffusionUpsample2d(conv(4,4)))),
      VaeUpBlock(Vector(res(4,2),res(2,2)),None)),norm(2),conv(2,3),Some(conv(1,1,1,padding=0)))
    (encoder,decoder)

  test("VAE image attention restores rectangular layout and channel means") {
    val width=2
    def linear(identity: Boolean)=Linear(Parameter(Tensor.fromFloats(
      if identity then Array(1f,0f,0f,1f) else Array.fill(4)(0f),Shape.from(Seq(2,2))),false),None)
    val attention=VaeAttention2d(GroupNorm.init(2,1,epsilon=.1,affine=false),
      DiffusionAttention(linear(false),linear(false),linear(true),linear(true),1))
    val values=Array.tabulate(24)(_.toFloat)
    val input=Variable(Tensor.fromFloats(values,Shape.from(Seq(2,2,2,3))),false)
    val result=attention(input).data.toFloatArray
    for b <- 0 until 2; c <- 0 until 2; i <- 0 until 6 do
      val mean=b*12+5.5
      val variance=(0 until 12).map(j => math.pow(j-5.5,2)).sum/12
      val channelMean=b*12+c*6+2.5
      val index=b*12+c*6+i
      assertEqualsDouble(result(index),values(index)+(channelMean-mean)/math.sqrt(variance+.1),2e-6)
  }
  test("encoder moments and decoder images differentiate every parameter and input") {
    val (encoder,decoder)=pair()
    val shape=Shape.from(Seq(1,3,6,10))
    val values=Array.tabulate(180)(i => math.sin(i*.13).toFloat)
    val x=Variable(Tensor.fromFloats(values,shape),true)
    val moments=encoder(x)
    assertEquals(moments.data.shape,Shape.from(Seq(1,2,3,5)))
    val image=decoder(moments.narrow(1,0,1))
    assertEquals(image.data.shape,shape)
    val imageSeed=Tensor.fromFloats(Array.tabulate(180)(i => math.cos(i*.21).toFloat),shape)
    val momentSeed=Tensor.fromFloats(Array.tabulate(30)(i => math.sin(i*.31).toFloat),moments.data.shape)
    val loss=(image*Variable(imageSeed,false)).sum()+(moments*Variable(momentSeed,false)).sum()
    loss.backward()
    val params=Module.parameters(encoder)++Module.parameters(decoder)
    assert(params.size>50)
    assert(params.forall(_.grad.exists(_.toFloatArray.forall(_.isFinite))))
    val originals=params.map(p => p->p.data).toMap
    val grads=params.map(p => p->p.grad.get.toFloatArray).toMap
    val dx=x.grad.get.toFloatArray
    for direction <- 0 until 3 do
      val random=new java.util.Random(1400+direction)
      def vector(n: Int)=Array.fill(n)((random.nextDouble()*.2-.1).toFloat)
      val xd=vector(values.length)
      val pd=params.map(p => p->vector(p.data.shape.numel.toInt)).toMap
      val expected=dx.zip(xd).map((a,b) => a.toDouble*b).sum+params.map(p => grads(p).zip(pd(p)).map((a,b) => a.toDouble*b).sum).sum
      def score(step: Float): Double = Autograd.noGrad {
        params.foreach(p => p.load(Tensor.fromFloats(originals(p).toFloatArray.zip(pd(p)).map((a,b) => a+step*b),p.data.shape)))
        val xx=Variable(Tensor.fromFloats(values.zip(xd).map((a,b) => a+step*b),shape),false)
        val mm=encoder(xx);val yy=decoder(mm.narrow(1,0,1))
        yy.data.toFloatArray.zip(imageSeed.toFloatArray).map((a,b) => a.toDouble*b).sum+
          mm.data.toFloatArray.zip(momentSeed.toFloatArray).map((a,b) => a.toDouble*b).sum
      }
      val numerical=(score(.002f)-score(-.002f))/.004
      params.foreach(p => p.load(originals(p)))
      assertEqualsDouble(expected,numerical,.003)
  }
  test("VAE frozen inference, odd encoder sizes and preflight limits") {
    val (encoder,decoder)=pair(false)
    def input(h: Int,w: Int)=Variable(Tensor.fromFloats(Array.fill(3*h*w)(.2f),Shape.from(Seq(1,3,h,w))),false)
    val moments=encoder(input(7,11))
    assertEquals(moments.data.shape,Shape.from(Seq(1,2,3,5)))
    val image=decoder(moments.narrow(1,0,1))
    assertEquals(image.data.shape,Shape.from(Seq(1,3,6,10)))
    assert(!moments.requiresGrad && !image.requiresGrad)
    intercept[IllegalArgumentException](encoder.copy(maxFeatureElements=1)(input(6,10)))
    intercept[IllegalArgumentException](decoder.copy(maxFeatureElements=1)(moments.narrow(1,0,1)))
    intercept[IllegalArgumentException](encoder(input(1,3)))
    intercept[IllegalArgumentException](encoder.copy(downBlocks=Vector.empty))
    intercept[IllegalArgumentException](decoder.copy(upBlocks=Vector.empty))
    intercept[IllegalArgumentException](decoder.copy(convIn=encoder.convIn))
    val limited=encoder.midBlock.attention.get.copy(attention=encoder.midBlock.attention.get.attention.copy(maxScoreElements=1))
    intercept[IllegalArgumentException](encoder.copy(midBlock=encoder.midBlock.copy(attention=Some(limited)))(input(6,10)))
  }
  test("VAE budgets include padding and upsampling intermediates, including empty-batch size bounds") {
    val (encoder,decoder)=pair(false)
    val image=Variable(Tensor.fromFloats(Array.fill(27)(.2f),Shape.from(Seq(1,3,3,3))),false)
    // Input is27 values but explicit bottom/right padding expands the hidden image to32.
    intercept[IllegalArgumentException](encoder.copy(maxFeatureElements=27)(image))
    val latent=Variable(Tensor.fromFloats(Array.fill(15)(.2f),Shape.from(Seq(1,1,3,5))),false)
    // Final RGB output is180 values; the preceding four-channel image needs240.
    intercept[IllegalArgumentException](decoder.copy(maxFeatureElements=180)(latent))
    assertEquals(decoder.copy(maxFeatureElements=240)(latent).data.shape,Shape.from(Seq(1,3,6,10)))
    val empty=Variable(Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,1,3,5))),false)
    assertEquals(decoder.copy(maxFeatureElements=0)(empty).data.shape,Shape.from(Seq(0,3,6,10)))
    val huge=Variable(Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,1,Int.MaxValue.toLong,1))),false)
    intercept[IllegalArgumentException](decoder(huge))
  }
  test("VAE rejects conditioning and malformed sampling/projections while optional branches run") {
    val (encoder,decoder)=pair(false)
    val block=encoder.downBlocks.head
    val conditioned=block.layers.head.copy(timeEmbProj=Some(Linear.init(3,2,42)))
    intercept[IllegalArgumentException](encoder.copy(downBlocks=encoder.downBlocks.updated(0,block.copy(layers=Vector(conditioned)))))
    intercept[IllegalArgumentException](encoder.copy(downBlocks=encoder.downBlocks.updated(0,block.copy(downsample=None))))
    intercept[IllegalArgumentException](encoder.copy(quantConv=decoder.postQuantConv))
    val input=Variable(Tensor.fromFloats(Array.fill(15)(.2f),Shape.from(Seq(1,1,3,5))),false)
    val noOptional=decoder.copy(postQuantConv=None,midBlock=decoder.midBlock.copy(attention=None))
    val image=noOptional(input)
    assertEquals(image.data.shape,Shape.from(Seq(1,3,6,10)))
    assert(!image.requiresGrad)
    assert(image.data.toFloatArray.forall(_.isFinite))
    val noQuant=encoder.copy(quantConv=None,midBlock=encoder.midBlock.copy(attention=None))
    assertEquals(noQuant(image).data.shape,Shape.from(Seq(1,2,3,5)))
  }
