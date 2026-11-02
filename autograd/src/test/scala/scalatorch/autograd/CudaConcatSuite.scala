package scalatorch.autograd

import scalatorch.*
import scalatorch.ffm.CudaSession

class CudaConcatSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def bits(x: Tensor[Float])=x.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits)
  test("resident concatenation matches CPU on every axis including repeated and empty inputs") {
    gpu()
    CudaSession.withDevice() { s =>
      for dims <- Seq(Seq(3L),Seq(2L,3L),Seq(2L,3L,2L,4L)); axis <- dims.indices do
        def input(width: Long,base: Float): Tensor[Float] =
          val shape=Shape.from(dims.updated(axis,width))
          Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => base+i),shape)
        val a=input(2,1f);val b=input(3,100f);val empty=input(0,0f)
        val cpu=Seq(a,empty,b,a)
        val da=s.upload(a);val db=s.upload(b);val de=s.upload(empty)
        val result=s.cat(Seq(da,de,db,da),axis)
        val expected=Variable.cat(cpu.map(Variable(_)),axis).data
        assertEquals(result.shape,expected.shape)
        assertEquals(bits(result.toCpu),bits(expected))
        da.close();db.close();de.close();result.close()
      val special=Tensor.fromFloats(Array(-0f,Float.PositiveInfinity,Float.NegativeInfinity,
        java.lang.Float.intBitsToFloat(0x7fc01234)),Shape.from(Seq(2L,2L))).transpose(0,1)
      val d=s.upload(special)
      val single=s.cat(Seq(d),0)
      d.close()
      assertEquals(bits(single.toCpu),bits(special))
      single.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("concatenation preflights invalid shapes, overflow, lifetime and device budget") {
    gpu()
    CudaSession.withDevice(maxBytes=40) { s =>
      val x=s.fill(Shape.from(Seq(2L,2L)),1f)
      intercept[IllegalArgumentException](s.cat(Seq.empty,0))
      intercept[IllegalArgumentException](s.cat(Seq(x),-1))
      intercept[IllegalArgumentException](s.cat(Seq(x),2))
      intercept[IllegalArgumentException](s.cat(Seq(x,x),1))
      val bad=s.fill(Shape.from(Seq(1L,3L)),0f)
      intercept[IllegalArgumentException](s.cat(Seq(x,bad),0))
      bad.close()
      assertEquals(s.allocatedBytes,16L)
      x.close()
      intercept[IllegalStateException](s.cat(Seq(x),0))
      val huge=s.fill(Shape.from(Seq(0L,Long.MaxValue,2L)),0f)
      val copy=s.cat(Seq(huge),1)
      assertEquals(copy.shape,huge.shape)
      copy.close()
      intercept[IllegalArgumentException](s.cat(Seq(huge,huge),1))
      huge.close()
      val empty=s.fill(Shape.from(Seq(2L,0L,3L)),0f)
      val result=s.cat(Seq(empty,empty),1)
      assertEquals(result.shape,empty.shape)
      assert(result.toCpu.toFloatArray.isEmpty)
      result.close();empty.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("upsampled features join a skip connection before resident normalization and convolution") {
    gpu()
    def data(dims: Seq[Long],seed: Int): Tensor[Float] =
      val shape=Shape.from(dims);val random=new java.util.Random(seed)
      Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
    val hidden=data(Seq(2L,2L,3L,4L),1)
    val skip=data(Seq(2L,4L,7L,9L),2)
    val weights=data(Seq(4L,6L,3L,3L),3)
    val config=Conv2dConfig(padding=(1,1))
    val joined=Variable.cat(Seq(Variable(hidden).resizeNearest2d(7,9),Variable(skip)),1)
    val expected=joined.groupNorm(2).conv2d(Variable(weights),config).data
    CudaSession.withDevice() { s =>
      val x=s.upload(hidden);val side=s.upload(skip);val w=s.upload(weights)
      val enlarged=s.resizeNearest2d(x,7,9)
      x.close()
      val merged=s.cat(Seq(enlarged,side),1)
      enlarged.close();side.close()
      val normalized=s.groupNorm(merged,2)
      merged.close()
      val result=s.conv2d(normalized,w,config)
      normalized.close();w.close()
      assertEquals(bits(result.toCpu),bits(expected))
      result.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
