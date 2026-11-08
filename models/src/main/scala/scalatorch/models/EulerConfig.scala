package scalatorch.models

import com.fasterxml.jackson.core.{JsonFactory,JsonToken,StreamReadConstraints,StreamReadFeature}
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import scala.collection.mutable
import scala.util.Using

/** Validated Euler checkpoint settings. Other scheduler families require explicit conversion. */
final class EulerConfig private (val noise: DiffusionNoiseSchedule,val prediction: DiffusionPrediction,
    val spacing: DiffusionTimestepSpacing,val stepsOffset: Int,val useKarrasSigmas: Boolean,
    val ddimClipSample: Boolean,val ddimClipRange: Float):
  def scheduler(steps: Int,maxElements: Long=134217728L): EulerScheduler =
    val schedule=if useKarrasSigmas then
      val karras=noise.karrasEuler(steps)
      if spacing==DiffusionTimestepSpacing.Leading then
        karras.copy(initialNoiseSigma=math.hypot(karras.initialNoiseSigma,1))
      else karras
    else noise.euler(steps,spacing,stepsOffset)
    EulerScheduler(schedule,prediction,maxElements)

object EulerConfig:
  private enum Value:
    case Number(n: BigDecimal)
    case Text(s: String)
    case Bool(b: Boolean)
    case Array(v: Vector[Value])
    case Null
  import Value.*
  private val limit=1048576
  private val factory: JsonFactory=JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
    .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(4).maxStringLength(limit).build()).build()
  def read(path: Path): EulerConfig = Using.resource(Files.newInputStream(path)) { input =>
    val bytes=input.readNBytes(limit+1)
    require(bytes.length<=limit,"Euler config exceeds 1MiB")
    fromJson(UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString)
  }
  def fromJson(json: String): EulerConfig =
    require(json.length<=limit,"Euler config exceeds size limit")
    try Using.resource(factory.createParser(json)) { p =>
      def value(arrayAllowed: Boolean=true): Value=p.currentToken() match
        case JsonToken.VALUE_NUMBER_INT | JsonToken.VALUE_NUMBER_FLOAT => Number(BigDecimal(p.getDecimalValue()))
        case JsonToken.VALUE_STRING => Text(p.getText())
        case JsonToken.VALUE_TRUE => Bool(true)
        case JsonToken.VALUE_FALSE => Bool(false)
        case JsonToken.VALUE_NULL => Null
        case JsonToken.START_ARRAY if arrayAllowed =>
          val values=Vector.newBuilder[Value]
          var count=0
          while p.nextToken()!=JsonToken.END_ARRAY do
            require(count<1000000,"Euler array too long")
            values+=value(false);count+=1
          Array(values.result())
        case _ => throw IllegalArgumentException("unsupported Euler config value")
      require(p.nextToken()==JsonToken.START_OBJECT,"Euler config must be an object")
      val fields=mutable.Map.empty[String,Value]
      while p.nextToken()!=JsonToken.END_OBJECT do
        require(p.currentToken()==JsonToken.FIELD_NAME && fields.size<128,"invalid Euler field")
        val key=p.currentName();p.nextToken();fields(key)=value()
      require(p.nextToken()==null,"trailing Euler config JSON")
      def take(key: String,default: Value=Null): Value=fields.remove(key).getOrElse(default)
      def exact(key: String,expected: Value): Unit=
        require(take(key,expected)==expected,s"unsupported Euler option: $key")
      def number(v: Value): Double=v match
        case Number(n) if n.toDouble.isFinite => n.toDouble
        case _ => throw IllegalArgumentException("expected finite Euler number")
      def integer(key: String,default: Int): Int=take(key,Number(BigDecimal(default))) match
        case Number(n) if n.isWhole && n>=Int.MinValue && n<=Int.MaxValue => n.toInt
        case _ => throw IllegalArgumentException(s"invalid Euler integer: $key")
      val sourceScheduler=take("_class_name") match
        case Null => ""
        case Text(name) if Set("EulerDiscreteScheduler","PNDMScheduler").contains(name) => name
        case Text(name) => throw IllegalArgumentException(s"unsupported Euler source scheduler: $name")
        case _ => throw IllegalArgumentException("invalid Euler source scheduler")
      for key <- Vector("_diffusers_version","_name_or_path") do
        take(key) match
          case Null | Text(_) => ()
          case _ => throw IllegalArgumentException(s"invalid Euler metadata: $key")
      take("_use_default_values") match
        case Null => ()
        case Array(values) => require(values.forall { case Text(_) => true; case _ => false },"invalid default metadata")
        case _ => throw IllegalArgumentException("invalid default metadata")
      if sourceScheduler=="PNDMScheduler" then
        for key <- Vector("set_alpha_to_one","skip_prk_steps") do
          take(key) match
            case Bool(_) => ()
            case _ => throw IllegalArgumentException(s"invalid PNDM metadata: $key")
      else
        // Diffusers retains fields from an earlier scheduler when saving a converted Euler config.
        for key <- Vector("set_alpha_to_one","skip_prk_steps") do
          take(key) match
            case Null | Bool(_) => ()
            case _ => throw IllegalArgumentException(s"invalid inherited scheduler metadata: $key")
      val ddimClipSample=take("clip_sample") match
        case Null => false
        case Bool(value) => value
        case _ => throw IllegalArgumentException("invalid clip_sample")
      val ddimClipRange=take("clip_sample_range") match
        case Null => 1f
        case Number(n) if n.toFloat.isFinite && n.toFloat>0 => n.toFloat
        case _ => throw IllegalArgumentException("invalid clip_sample_range")
      take("sample_max_value") match
        case Null => ()
        case Number(n) if n.toDouble.isFinite => ()
        case _ => throw IllegalArgumentException("invalid inherited scheduler metadata: sample_max_value")
      val karras=take("use_karras_sigmas",Bool(false)) match
        case Bool(b) => b
        case _ => throw IllegalArgumentException("invalid Karras flag")
      val zeroSnr=take("rescale_betas_zero_snr",Bool(false)) match
        case Bool(b) => b
        case _ => throw IllegalArgumentException("invalid rescale_betas_zero_snr flag")
      for key <- Vector("use_exponential_sigmas","use_beta_sigmas") do
        exact(key,Bool(false))
      exact("interpolation_type",Text("linear"));exact("timestep_type",Text("discrete"))
      exact("final_sigmas_type",Text("zero"));exact("sigma_min",Null);exact("sigma_max",Null)
      val prediction=take("prediction_type",Text("epsilon")) match
        case Text("epsilon") => DiffusionPrediction.Epsilon
        case Text("sample") => DiffusionPrediction.Sample
        case Text("v_prediction") => DiffusionPrediction.VPrediction
        case _ => throw IllegalArgumentException("unsupported Euler prediction")
      val spacing=take("timestep_spacing",Text("linspace")) match
        case Text("linspace") => DiffusionTimestepSpacing.Linspace
        case Text("leading") => DiffusionTimestepSpacing.Leading
        case Text("trailing") => DiffusionTimestepSpacing.Trailing
        case _ => throw IllegalArgumentException("unsupported Euler spacing")
      val count=integer("num_train_timesteps",1000)
      require(count>0 && count<=1000000,"invalid Euler training steps")
      val offset=integer("steps_offset",0)
      val start=number(take("beta_start",Number(BigDecimal("0.0001"))))
      val end=number(take("beta_end",Number(BigDecimal("0.02"))))
      val betaSchedule=take("beta_schedule",Text("linear"))
      val rawNoise=take("trained_betas") match
        case Array(values) =>
          require(values.size==count,"trained betas length differs from training steps")
          DiffusionNoiseSchedule.fromBetas(values.map(number))
        case Null =>
          require(betaSchedule==Text("linear") || betaSchedule==Text("scaled_linear"),"unsupported Euler beta schedule")
          DiffusionNoiseSchedule.linear(count,start,end,betaSchedule==Text("scaled_linear"))
        case _ => throw IllegalArgumentException("invalid trained betas")
      require(fields.isEmpty,s"unknown Euler config fields: ${fields.keys.toVector.sorted.mkString(", ")}")
      val noise=if zeroSnr then rawNoise.withZeroTerminalSnr else rawNoise
      new EulerConfig(noise,prediction,spacing,offset,karras,ddimClipSample,ddimClipRange)
    }
    catch case e: IOException => throw IllegalArgumentException("invalid Euler config JSON",e)
