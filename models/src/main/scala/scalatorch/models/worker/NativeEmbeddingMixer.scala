package scalatorch.models.worker

/** Float32 mixing with aligned SDXL row filtering. */
private[worker] object NativeEmbeddingMixer:
  type Vectors=Map[String,Vector[Vector[Float]]]
  def mix(inputs: Vector[(Vectors,Double)],concat: Boolean,combine: Boolean,expression: Option[String]=None): Vectors =
    val preset=expression.map(_.trim).filter(_.nonEmpty)
    val supported=Set("=v*8","=math.ceil(v*8)/8","=(1*(v>=0)-1*(v<0))/50","=v*random.random()",
      "=v*math.sin(i/maxi*math.pi)","=v*((i%2)==0)","=v*(i<maxi//2)","=v*(i>=maxi//2)")
    require(preset.forall(supported),"unsupported native embedding expression")
    require(inputs.forall((_,weight)=>weight.isFinite),"weight must be finite")
    val active=inputs.filter(_._2!=0.0)
    require(active.nonEmpty,"no embeddings to mix")
    val keys=active.head._1.keySet
    require(keys==Set("clip_l") || keys==Set("clip_l","clip_g"),"unsupported encoder set")
    require(active.forall(_._1.keySet==keys),"embedding encoder set mismatch")
    for (vectors,_) <- active do
      require(vectors.values.forall(rows=>rows.nonEmpty && rows.head.nonEmpty &&
        rows.forall(row=>row.size==rows.head.size && row.forall(_.isFinite))),"invalid embedding vectors")
      require(!keys.contains("clip_g") || vectors("clip_l").size==vectors("clip_g").size,
        "embedding encoder vector counts differ")
    val result=keys.toVector.sorted.map { key =>
      val width=active.head._1(key).head.size
      require(active.forall(_._1(key).head.size==width),"embedding dimension mismatch")
      val weighted=active.map { (values,weight) =>
        values(key).map(_.map(value=>(value*weight.toFloat).toFloat))
      }
      val rows=if concat then weighted.flatten else
        val count=weighted.map(_.size).max
        Vector.tabulate(count) { row => Vector.tabulate(width) { column =>
          weighted.foldLeft(0f)((sum,vectors)=>if row<vectors.size then (sum+vectors(row)(column)).toFloat else sum)
        }}
      val transformed=rows.map { row => row.zipWithIndex.map { (v,i) =>
        preset match
          case None => v
          case Some("=v*8") => (v*8f).toFloat
          case Some("=math.ceil(v*8)/8") => (math.ceil((v*8f).toDouble)/8.0).toFloat
          case Some("=(1*(v>=0)-1*(v<0))/50") => if v>=0 then 0.02f else -0.02f
          case Some("=v*random.random()") => (v*java.util.concurrent.ThreadLocalRandom.current().nextDouble().toFloat).toFloat
          case Some("=v*math.sin(i/maxi*math.pi)") => (v*math.sin(i.toDouble/width*math.Pi).toFloat).toFloat
          case Some("=v*((i%2)==0)") => if i%2==0 then v else v*0f
          case Some("=v*(i<maxi//2)") => if i<width/2 then v else v*0f
          case Some("=v*(i>=maxi//2)") => if i>=width/2 then v else v*0f
          case _ => throw IllegalArgumentException("unsupported native embedding expression")
      }}
      val output=if combine && transformed.size>1 then
        Vector(Vector.tabulate(width)(column=>transformed.foldLeft(0f)((sum,row)=>(sum+row(column)).toFloat)))
      else transformed
      require(output.forall(_.forall(_.isFinite)),"mixed embedding contains non-finite values")
      key->output
    }.toMap
    val keep=result("clip_l").indices.filter(index=>keys.exists(key=>result(key)(index).exists(_!=0f))).toVector
    require(keep.nonEmpty,"all mixed vectors are zero")
    result.map((key,rows)=>key->keep.map(rows(_)))
