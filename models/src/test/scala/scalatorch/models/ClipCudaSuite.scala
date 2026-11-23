package scalatorch.models

import munit.FunSuite
import scalatorch.*
import scalatorch.autograd.Autograd
import scalatorch.nn.*

class ClipCudaSuite extends FunSuite:
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA model gate disabled")
  private def param(random: java.util.Random,shape: Long*): Parameter =
    Parameter(Tensor.fromFloats(Array.fill(shape.product.toInt)((random.nextGaussian()*0.08).toFloat),Shape.from(shape)),false)
  private def ones(width: Int,random: java.util.Random) =
    LayerNorm(width,Some(Parameter(Tensor.fromFloats(Array.fill(width)((1+random.nextGaussian()*0.05).toFloat),Shape.from(Seq(width.toLong))),false)),
      Some(param(random,width.toLong)))
  private def model(activation: ClipActivation): ClipTextModel =
    val r=new java.util.Random(5);val w=64;val h=4
    def lin(o: Long,i: Long)=Linear(param(r,o,i),Some(param(r,o)))
    def layer=ClipTextLayer(ones(w,r),lin(w,w),lin(w,w),lin(w,w),lin(w,w),ones(w,r),lin(128,w),lin(w,128),h,activation)
    ClipTextModel(Embedding(param(r,100,w)),Embedding(param(r,77,w)),Vector(layer,layer,layer),ones(w,r),
      textProjection=Some(Linear(param(r,32,w),None)),eosTokenId=99L)
  for activation <- Seq(ClipActivation.QuickGelu,ClipActivation.Gelu) do
    test(s"CUDA CLIP layers match the CPU reference with $activation") {
      gpu()
      val m=model(activation)
      val tokens=Tensor.fromLongs(Array.tabulate(2*20)(i => if i%20==19 then 99L else (i*7%90).toLong),Shape.from(Seq(2L,20L)))
      ClipCuda.disable()
      val cpu=Autograd.noGrad(m.encode(tokens,skipLastLayers=1))
      ClipCuda.enable(0)
      assert(ClipCuda.run(m,Tensor.fromFloats(Array.fill(1*4*64)(0.1f),Shape.from(Seq(1L,4L,64L))),1).isDefined,"CUDA CLIP path did not run")
      val cuda=try Autograd.noGrad(m.encode(tokens,skipLastLayers=1)) finally ClipCuda.disable()
      // Float16 is the CUDA default, so the CPU reference only matches to half precision.
      val tolerance=if CudaDiffusionRuntime.fp16Default then 1e-2 else 2e-4
      def close(a: Tensor[Float],b: Tensor[Float]) =
        a.toFloatArray.zip(b.toFloatArray).foreach { (x,y) => assert(math.abs(x-y)<=tolerance*(1+math.abs(y)),s"$x vs $y") }
      close(cuda.hidden.data,cpu.hidden.data);close(cuda.finalHidden.data,cpu.finalHidden.data)
      close(cuda.pooled.data,cpu.pooled.data)
    }
