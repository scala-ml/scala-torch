package scalatorch.models.worker

/** Per-channel byte histogram quantile matching, compatible with scikit-image. */
private[worker] object NativeHistogramMatch:
  def rgbChannels(source: Array[Int],reference: Array[Int],cancelled: () => Boolean): Array[Int] =
    require(source.nonEmpty && reference.nonEmpty && source.length%3==0 && reference.length%3==0,
      "histogram images must contain RGB triples")
    require(source.forall(v=>v>=0 && v<=255) && reference.forall(v=>v>=0 && v<=255),"histogram samples must be bytes")
    val output=new Array[Int](source.length)
    for channel <- 0 until 3 do
      val sourceCounts=new Array[Int](256);val referenceCounts=new Array[Int](256)
      var i=channel
      while i<source.length do
        if (i&65535)==channel && (cancelled() || Thread.currentThread().isInterrupted) then
          throw new java.util.concurrent.CancellationException("histogram matching cancelled")
        sourceCounts(source(i))+=1;i+=3
      i=channel
      while i<reference.length do
        if (i&65535)==channel && (cancelled() || Thread.currentThread().isInterrupted) then
          throw new java.util.concurrent.CancellationException("histogram matching cancelled")
        referenceCounts(reference(i))+=1;i+=3
      val referenceValues=(0 until 256).filter(referenceCounts(_)>0).toArray
      val referenceQuantiles=new Array[Double](referenceValues.length)
      var cumulative=0L
      for j <- referenceValues.indices do
        cumulative+=referenceCounts(referenceValues(j))
        referenceQuantiles(j)=cumulative.toDouble/(reference.length/3)
      val mapping=new Array[Int](256)
      var sourceCumulative=0L;var k=0
      for value <- 0 until 256 if sourceCounts(value)>0 do
        sourceCumulative+=sourceCounts(value)
        val q=sourceCumulative.toDouble/(source.length/3)
        while k<referenceQuantiles.length-1 && referenceQuantiles(k)<q do k+=1
        val mapped=if k==0 then referenceValues(0).toDouble
          else if referenceQuantiles(k)==q then referenceValues(k).toDouble
          else
            val slope=(referenceValues(k)-referenceValues(k-1)).toDouble/(referenceQuantiles(k)-referenceQuantiles(k-1))
            slope*(q-referenceQuantiles(k-1))+referenceValues(k-1)
        mapping(value)=mapped.toInt
      i=channel
      while i<source.length do
        if (i&65535)==channel && (cancelled() || Thread.currentThread().isInterrupted) then
          throw new java.util.concurrent.CancellationException("histogram matching cancelled")
        output(i)=mapping(source(i));i+=3
    output
