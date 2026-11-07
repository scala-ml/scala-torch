package scalatorch.models

import scalatorch.Tensor
import scala.collection.mutable.ArrayDeque

/** Request-local history for NUI Anti-burn's early-stop/average callback. */
final case class AntiBurn(skipLast: Int=2,average: Int=3):
  require(skipLast>=0 && skipLast<=50 && average>=1 && average<=50,"invalid anti-burn options")
  private[models] def newState(): AntiBurn.State=new AntiBurn.State(this)

object AntiBurn:
  private[models] final class State(settings: AntiBurn):
    private val history=ArrayDeque.empty[Array[Float]]
    def step(current: Tensor[Float],completed: Int,total: Int): (Tensor[Float],Boolean) =
      history.append(current.toFloatArray)
      while history.size>settings.average do history.removeHead()
      val stop=settings.skipLast>0 && completed>=total-settings.skipLast
      if stop || (settings.skipLast==0 && completed==total) then
        val out=new Array[Float](history.head.length)
        for values <- history do
          var i=0
          while i<out.length do {out(i)+=values(i);i+=1}
        val count=history.size.toFloat
        var i=0
        while i<out.length do {out(i)/=count;i+=1}
        (Tensor.fromFloats(out,current.shape),stop)
      else (current,false)
