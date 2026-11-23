package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import scalatorch.models.{OpenPoseBody,OpenPoseDetection}

/** The `openpose` ControlNet preprocessor: the CMU OpenPose COCO body network on the CPU rendered as a ControlNet stick figure.
  * There is no fallback when the weights are missing. The weights (`body_pose_model.pth`) are CMU OpenPose's and are licensed
  * for academic / non-commercial use only. */
object NativeOpenPose:
  val FileName="body_pose_model.pth"
  val StickWidth=4
  private var cached: Option[(Path,OpenPoseBody)]=None

  /** Standard OpenPose colours, one per keypoint and (for the first 17) per limb. */
  val Colors=Vector((255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),
    (0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85))

  /** Candidate weight files: SDFX_SCALATORCH_ANNOTATOR_DIR, then models/annotators in the working directory and its parents. */
  def candidates(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Vector[Path]=
    val fromEnv=env.get("SDFX_SCALATORCH_ANNOTATOR_DIR").filter(_.nonEmpty).toVector.map(d=>Path.of(d).resolve(FileName))
    fromEnv++Iterator.iterate(cwd)(_.getParent).takeWhile(_!=null).map(_.resolve("models").resolve("annotators").resolve(FileName)).toVector

  def locate(env: Map[String,String]=sys.env,cwd: Path=Path.of("").toAbsolutePath): Path=
    val options=candidates(env,cwd)
    options.find(Files.isRegularFile(_)).getOrElse(
      throw IllegalStateException(s"OpenPose body weights not found (expected $FileName); set SDFX_SCALATORCH_ANNOTATOR_DIR or place them under models/annotators. Searched: ${options.take(4).mkString(", ")}"))

  private def model(file: Path,cancelled: () => Boolean): OpenPoseBody=synchronized {
    cached match
      case Some((p,m)) if p==file => m
      case _ => val m=OpenPoseBody.load(file,cancelled);cached=Some((file,m));m
  }

  /** BGR HWC 0..255 samples of an image. */
  def bgr(image: BufferedImage): Array[Float]=
    val w=image.getWidth;val h=image.getHeight;val out=new Array[Float](3*w*h)
    for y <- 0 until h; x <- 0 until w do
      val c=image.getRGB(x,y);val o=(y*w+x)*3
      out(o)=(c&255).toFloat;out(o+1)=((c>>>8)&255).toFloat;out(o+2)=((c>>>16)&255).toFloat
    out

  def detect(source: BufferedImage,weights: Path,cancelled: () => Boolean): OpenPoseDetection=
    OpenPoseBody.detect(model(weights,cancelled),bgr(source),source.getHeight,source.getWidth,cancelled)

  /** Draws the ControlNet pose map (black background, limb ellipses blended at 60%, filled keypoint discs of radius 4). */
  def render(detection: OpenPoseDetection): BufferedImage=
    val w=detection.width;val h=detection.height
    val pixels=new Array[Int](w*h)
    java.util.Arrays.fill(pixels,0xff000000.toInt)
    val byId=detection.peaks.flatten.sortBy(_.id).toArray
    for i <- 0 until 17; person <- detection.people do
      val (ia,ib)=OpenPoseBody.LimbSeq(i);val pa=person(ia-1).toInt;val pb=person(ib-1).toInt
      if pa>=0 && pb>=0 then
        val (ax,ay)=(byId(pa).x.toDouble,byId(pa).y.toDouble);val (bx,by)=(byId(pb).x.toDouble,byId(pb).y.toDouble)
        val length=math.hypot(ax-bx,ay-by);val angle=math.toDegrees(math.atan2(ay-by,ax-bx)).toInt
        val cx=((ax+bx)/2).toInt;val cy=((ay+by)/2).toInt
        val semi=(length/2).toInt;val rot=math.toRadians(angle)
        val polygon=new java.awt.Polygon
        for deg <- 0 until 360 do
          val t=math.toRadians(deg);val ex=semi*math.cos(t);val ey=StickWidth*math.sin(t)
          polygon.addPoint(math.rint(cx+ex*math.cos(rot)-ey*math.sin(rot)).toInt,math.rint(cy+ex*math.sin(rot)+ey*math.cos(rot)).toInt)
        val box=polygon.getBounds;val (r,g,b)=Colors(i)
        for y <- math.max(0,box.y) to math.min(h-1,box.y+box.height); x <- math.max(0,box.x) to math.min(w-1,box.x+box.width) do
          if polygon.contains(x,y) then
            val old=pixels(y*w+x)
            def mix(o: Int,c: Int)=math.round(0.4*o+0.6*c).toInt
            pixels(y*w+x)=0xff000000.toInt|(mix((old>>>16)&255,r)<<16)|(mix((old>>>8)&255,g)<<8)|mix(old&255,b)
    for i <- 0 until 18; person <- detection.people do
      val id=person(i).toInt
      if id>=0 then
        val p=byId(id);val (r,g,b)=Colors(i);val rgb=0xff000000.toInt|(r<<16)|(g<<8)|b
        for dy <- -4 to 4; dx <- -4 to 4 do
          val x=p.x+dx;val y=p.y+dy
          if dx*dx+dy*dy<=16 && x>=0 && x<w && y>=0 && y<h then pixels(y*w+x)=rgb
    val image=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    image.setRGB(0,0,w,h,pixels,0,w)
    image

  /** Detects at `resolution` pixels on the shorter side (as the ControlNet annotators do; 0 keeps the source size), draws the pose
    * and resizes the map back to the source size. A map with no people is plain black. */
  def preprocess(source: BufferedImage,resolution: Int,cancelled: () => Boolean,env: Map[String,String]=sys.env): BufferedImage=
    val weights=locate(env)
    val shorter=math.min(source.getWidth,source.getHeight)
    val working=if resolution>0 && resolution!=shorter then
      val s=resolution.toDouble/shorter
      NativeControlNetImageOps.bicubic(source,math.max(1,math.round(source.getWidth*s).toInt),math.max(1,math.round(source.getHeight*s).toInt),cancelled)
      else source
    val pose=render(detect(working,weights,cancelled))
    if pose.getWidth==source.getWidth && pose.getHeight==source.getHeight then pose
    else NativeControlNetImageOps.bicubic(pose,source.getWidth,source.getHeight,cancelled)
