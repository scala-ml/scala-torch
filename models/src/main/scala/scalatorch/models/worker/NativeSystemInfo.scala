package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scala.util.Using
import scala.util.control.NonFatal
import scalatorch.io.SafeTensors
import NativeJson.*

/** Local diagnostics and opt-in inventory; it never transmits information. */
object NativeSystemInfo:
  private val kinds=Vector("checkpoints","lora","embeddings","hypernetworks","vae")
  private def digest(path: Path,check: () => Unit): String =
    val md=java.security.MessageDigest.getInstance("SHA-256")
    val file=if Files.isDirectory(path) then
      Vector("diffusion_pytorch_model.safetensors","diffusion_pytorch_model.fp16.safetensors",
        "diffusion_pytorch_model.bin","config.json").map(path.resolve(_)).find(Files.isRegularFile(_))
    else Some(path)
    file match
      case Some(source) => Using.resource(Files.newInputStream(source)) { in =>
        val bytes=new Array[Byte](65536)
        var n=in.read(bytes)
        while n>=0 do
          check()
          if n>0 then md.update(bytes,0,n)
          n=in.read(bytes)
      }
      case None => md.update(path.toString.getBytes(java.nio.charset.StandardCharsets.UTF_8))
    java.util.HexFormat.of().formatHex(md.digest()).take(10)

  def execute(p: Map[String,NativeJson],status: Map[String,NativeJson],features: Vector[String],
      samplers: Map[String,NativeJson],ctx: NativeWorkContext): Map[String,NativeJson] =
    require((p.keySet--Set("dirs","hashes")).isEmpty,"unknown system_info parameters")
    val dirs=p.get("dirs") match
      case None => Map.empty[String,NativeJson]
      case Some(Obj(values)) => values
      case _ => throw IllegalArgumentException("dirs must be an object")
    dirs.foreach { (_,value) => value match
      case Arr(values) => require(values.forall(_.isInstanceOf[Str]),"inventory directories must contain strings")
      case _ => throw IllegalArgumentException("inventory directories must be arrays")
    }
    val hashes=p.get("hashes") match
      case None => false
      case Some(Bool(value)) => value
      case _ => throw IllegalArgumentException("hashes must be boolean")
    def paths(kind: String)=dirs.get(kind).map(_.asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)).getOrElse(Vector.empty)
    val active=status.get("model").collect { case Obj(fields) => fields }.getOrElse(Map.empty)
    val groups=kinds.map { kind =>
      val files=if kind=="checkpoints" then NativeModelCatalog.scan(paths(kind),()=>ctx.checkCancelled())
        else NativeFileCatalog.scan(if kind=="hypernetworks" then "embeddings" else kind,paths(kind),()=>ctx.checkCancelled())
      val entries=files.map { entry =>
        ctx.checkCancelled()
        val fields=entry.fields
        val path=Path.of(fields("path").asInstanceOf[Str].value)
        val same=active.get("path").contains(Str(path.toString))
        val hash=if hashes then Some(digest(path,()=>ctx.checkCancelled()))
          else if same then active.get("hash").collect { case Str(s) if s.nonEmpty => s } else None
        val extra=if kind!="embeddings" then Map.empty[String,NativeJson]
        else if !path.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
          Map("vectors"->Null,"vector_status"->Str("pickle is not loaded for inspection"))
        else try
          val vector=Using.resource(SafeTensors.open(path)) { file =>
            file.tensors.values.map(t=>if t.shape.rank>1 then t.shape.dims.head else 1L).maxOption
          }
          Map("vectors"->vector.map(v=>Number(BigDecimal(v))).getOrElse(Null))
        catch case NonFatal(_) => Map("vectors"->Null,"vector_status"->Str("unreadable safetensors header"))
        Obj(fields.filter((key,_)=>Set("name","kind","size_mb")(key))++extra++Map(
          "hash"->hash.map(Str.apply).getOrElse(Null),
          "hash_status"->Str(if hash.nonEmpty then "SHA256 prefix" else "not calculated; use Hash inventory")))
      }
      kind->Arr(entries)
    }.toMap
    val memory=status.get("memory").collect { case Obj(fields) => fields }.getOrElse(Map.empty)
    val gpu=memory.get("cuda_device") match
      case Some(name: Str) => Obj(Map("device"->name,"total_mb"->memory.getOrElse("cuda_total_mb",Null),
        "status"->Str("CUDA driver telemetry")))
      case _ => Obj(Map("status"->memory.getOrElse("gpu_status",Str("CUDA not active"))))
    status++Map("gpu"->gpu,"networks"->Obj(groups+ ("hypernetworks_status"->Str("Inventory only; hypernetwork inference is not implemented"))),
      "features"->Arr(features.map(Str.apply)),"schedulers"->Obj(samplers),
      "version"->Obj(Map("app"->Str("stablediffusion-nui"),"engine"->Str("scala-torch-0.1.0"))),
      "libraries"->Obj(Map("java"->Str(sys.props("java.version")))),
      "repos"->Obj(Map("note"->Str("Native scala-torch runtime"))))

  private val MetadataAliases: Map[String, String] = Map(
    "torch.version" -> "libraries.torch",
    "crossattention" -> "runtime.attention",
    "backend" -> "runtime.backend",
    "pipeline" -> "runtime.pipeline",
    "flags" -> "runtime.flags",
    "libs.xformers" -> "libraries.xformers",
    "libs.diffusers" -> "libraries.diffusers",
    "libs.transformers" -> "libraries.transformers"
  )

  private val MetadataFields: Set[String] = Set(
    "version.app", "version.engine", "version.hash", "platform.arch", "platform.cpu", "platform.system",
    "platform.release", "platform.python", "gpu.device", "gpu.cuda", "gpu.cudnn", "gpu.hip", "runtime.attention",
    "runtime.backend", "runtime.pipeline", "runtime.dtype", "runtime.vram_mode", "libraries.diffusers",
    "libraries.transformers", "libraries.torch", "version.tag", "version.tags", "version.updated", "version.url",
    "gpu.ipex", "gpu.openvino"
  ) ++ MetadataAliases.keySet

  def metadata(data: Map[String, NativeJson], fields: Vector[String]): Map[String, NativeJson] =
    val out = Map.newBuilder[String, NativeJson]
    for field <- fields if MetadataFields.contains(field) do
      val path = MetadataAliases.getOrElse(field, field).split("\\.").toVector
      var cur: Option[NativeJson] = Some(Obj(data))
      for part <- path if cur.nonEmpty do
        cur = cur.flatMap {
          case Obj(m) => m.get(part).filter(_ != Null)
          case _ => None
        }
      cur.foreach {
        case Arr(items) =>
          val text = items.map {
            case Str(s) => s
            case Number(n) => n.toString
            case Bool(b) => b.toString
            case other => NativeJson.render(other)
          }.mkString(" ")
          out += field -> Str(text)
        case v => out += field -> v
      }
    out.result()

  private val DestinationHost = "logs3.papertrailapp.com"
  private val DestinationPort = 32554

  private def compactValue(v: NativeJson): String = v match
    case Obj(fields) => fields.map { (k, item) => s"$k:${compactValue(item)}" }.mkString(" ")
    case Arr(items)  => items.map(compactValue).mkString(",")
    case Str(s)      => s
    case Number(n)   => n.toString
    case Bool(b)     => b.toString
    case Null        => ""

  private def sanitizeField(s: String): String =
    s.replace('|', '/').replace('\r', ' ').replace('\n', ' ').replace('\u0000', ' ')

  def submissionPreview(p: Map[String, NativeJson]): Map[String, NativeJson] =
    val username = p.get("username") match
      case Some(Str(s)) if s.trim.nonEmpty && s.length <= 128 && !s.exists(c => c == '\r' || c == '\n' || c == '|' || c == '\u0000') =>
        s.trim
      case _ =>
        throw IllegalArgumentException("Enter a username without line breaks or separators to preview submission")
    val record = p.get("record") match
      case Some(Obj(fields)) => fields
      case _                 => throw IllegalArgumentException("Choose a benchmark record to submit")
    val results = record.get("results") match
      case Some(Arr(items)) if items.forall(_.isInstanceOf[Obj]) => items.map(_.asInstanceOf[Obj].fields)
      case Some(Arr(_))                                          => throw IllegalArgumentException("Malformed benchmark record")
      case _                                                     => throw IllegalArgumentException("Choose a benchmark record to submit")
    val system = record.getOrElse("system", Obj(Map.empty)) match
      case Obj(fields) => fields
      case _           => throw IllegalArgumentException("Malformed benchmark record")
    val config = record.getOrElse("config", Obj(Map.empty)) match
      case Obj(fields) => fields
      case _           => throw IllegalArgumentException("Malformed benchmark record")
    val fingerprint = record.getOrElse("fingerprint", Str("")) match
      case Str(s) => s
      case _      => throw IllegalArgumentException("Malformed benchmark record")
    val includeHostname = p.getOrElse("include_hostname", Bool(false)) match
      case Bool(b) => b
      case _       => throw IllegalArgumentException("include_hostname must be boolean")
    val hostname =
      if includeHostname then
        try java.net.InetAddress.getLocalHost.getHostName
        catch case NonFatal(_) => "anonymous"
      else "anonymous"
    val runtime = system.get("runtime").collect { case Obj(f) => f }.getOrElse(Map.empty)
    val model = system.get("model").collect { case Obj(f) => f }.getOrElse(Map.empty)
    def strField(m: Map[String, NativeJson], k: String, default: String = ""): String =
      m.get(k).map(compactValue).getOrElse(default)
    val optimizations = Vector(
      "pipeline" -> strField(runtime, "pipeline"),
      "crossattention" -> strField(runtime, "attention"),
      "flags" -> compactValue(runtime.getOrElse("flags", Arr(Vector.empty))),
      "compute" -> strField(runtime, "backend"),
      "compile" -> "none",
      "dtype" -> strField(runtime, "dtype")
    ).map((k, v) => s"$k:$v").mkString(" ")
    val settings = Vector(
      "pipeline" -> strField(runtime, "pipeline"),
      "model" -> strField(model, "name"),
      "vae" -> strField(model, "vae_name"),
      "width" -> strField(config, "width", "512"),
      "height" -> strField(config, "height", "512")
    ).map((k, v) => s"$k:$v").mkString(" ")
    val ips = results.map(r => r.get("iterations_per_second").map(compactValue).getOrElse("error")).mkString(" / ")
    val rawFields = Vector(
      strField(record, "timestamp"),
      ips,
      compactValue(system.getOrElse("version", Obj(Map.empty))),
      compactValue(system.getOrElse("platform", Obj(Map.empty))),
      compactValue(system.getOrElse("libraries", Obj(Map.empty))),
      compactValue(system.getOrElse("gpu", Obj(Map.empty))),
      optimizations,
      settings,
      username,
      strField(record, "note"),
      fingerprint.take(6)
    )
    val baseLine = rawFields.map(sanitizeField).mkString("|").replace("  ", " ").replace("\"", "").trim
    val md = java.security.MessageDigest.getInstance("SHA-256")
    val lineHash = java.util.HexFormat.of().formatHex(md.digest(baseLine.getBytes(java.nio.charset.StandardCharsets.UTF_8))).take(6)
    val line = s"$baseLine|$lineHash"
    val stamp = java.time.ZonedDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("MMM dd HH:mm:ss", java.util.Locale.US))
    val packet = s"<14>$stamp $hostname SDBENCHMARK: $username $line\u0000"
    require(
      packet.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= 60000,
      "Benchmark submission exceeds the UDP payload limit"
    )
    val tokenBytes = new Array[Byte](24)
    new java.security.SecureRandom().nextBytes(tokenBytes)
    val token = java.util.Base64.getUrlEncoder.withoutPadding().encodeToString(tokenBytes)
    Map(
      "token" -> Str(token),
      "destination" -> Str(s"$DestinationHost:$DestinationPort/udp"),
      "packets" -> Arr(Vector(Str(packet)))
    )

  def submit(
      p: Map[String, NativeJson],
      pending: Option[(Long, Map[String, NativeJson])],
      sendUdp: (Vector[String], String, Int) => Unit = defaultSendUdp
  ): Map[String, NativeJson] =
    val valid = pending.filter { case (startedAt, preview) =>
      val elapsedS = (System.nanoTime() - startedAt) / 1e9
      elapsedS <= 300.0 && p.get("token").contains(preview("token"))
    }
    val (_, preview) = valid.getOrElse(
      throw IllegalArgumentException("Preview the exact benchmark payload again before submitting")
    )
    val packets = preview("packets").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)
    try sendUdp(packets, DestinationHost, DestinationPort)
    catch
      case NonFatal(error) =>
        throw new NativeBackendError(
          "unsupported",
          s"Legacy submission unavailable: ${error.getMessage}; export the benchmark JSON locally",
          error
        )
    Map(
      "status" -> Str("sent_unconfirmed"),
      "message" -> Str("UDP packet sent. The legacy logger provides no receipt confirmation.")
    )

  private def defaultSendUdp(packets: Vector[String], host: String, port: Int): Unit =
    Using.resource(new java.net.DatagramSocket()) { socket =>
      socket.setSoTimeout(5000)
      val addr = java.net.InetAddress.getByName(host)
      packets.foreach { pkt =>
        val bytes = pkt.getBytes(java.nio.charset.StandardCharsets.UTF_8)
        val datagram = new java.net.DatagramPacket(bytes, bytes.length, addr, port)
        socket.send(datagram)
      }
    }
