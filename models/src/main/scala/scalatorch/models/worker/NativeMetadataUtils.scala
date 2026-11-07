package scalatorch.models.worker

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.util.{HexFormat, UUID}
import scala.util.Using
import scala.util.control.NonFatal

private[worker] object NativeMetadataUtils:
  import NativeJson.*

  private val MaxHeader = 100L * 1024L * 1024L
  private val MaxMetadata = 4 * 1024 * 1024
  private val Chunk = 1024 * 1024
  private val Bits: Map[String, Int] = Map(
    "BOOL" -> 8, "U8" -> 8, "I8" -> 8, "I16" -> 16, "U16" -> 16, "F16" -> 16, "BF16" -> 16,
    "I32" -> 32, "U32" -> 32, "F32" -> 32, "F64" -> 64, "I64" -> 64, "U64" -> 64,
    "F8_E4M3" -> 8, "F8_E5M2" -> 8, "F8_E8M0" -> 8, "F8_E4M3FNUZ" -> 8, "F8_E5M2FNUZ" -> 8,
    "C64" -> 64, "F4" -> 4, "F6_E2M3" -> 6, "F6_E3M2" -> 6
  )

  private final case class FileIdentity(size: Long, mtimeNs: Long, ctimeNs: Long, keyHash: Long):
    def toJson: Arr = Arr(Vector(Number(0), Number(BigDecimal(keyHash)), Number(BigDecimal(size)),
      Number(BigDecimal(mtimeNs)), Number(BigDecimal(ctimeNs))))

  private def identityOf(path: Path): FileIdentity =
    val attrs = Files.readAttributes(path, classOf[BasicFileAttributes])
    val mtime = attrs.lastModifiedTime().toInstant
    val ctime = attrs.creationTime().toInstant
    val mtimeNs = mtime.getEpochSecond * 1000000000L + mtime.getNano
    val ctimeNs = ctime.getEpochSecond * 1000000000L + ctime.getNano
    val keyHash = Option(attrs.fileKey()).map(_.toString.hashCode.toLong & 0xffffffffL).getOrElse(0L)
    FileIdentity(attrs.size(), mtimeNs, ctimeNs, keyHash)

  private def ensureUnchanged(path: Path, expected: FileIdentity): Unit =
    if identityOf(path) != expected then
      throw IllegalArgumentException("Source changed; reload metadata before saving.")

  private def resolveSource(raw: String): Path =
    require(raw.nonEmpty && raw.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors"),
      "Choose an existing .safetensors file")
    val p = Path.of(raw)
    require(Files.isRegularFile(p), "Choose an existing .safetensors file")
    p.toRealPath()

  private def parseHeader(in: java.io.InputStream, fileSize: Long): (Map[String, NativeJson], Long, Array[Byte]) =
    val lenBytes = in.readNBytes(8)
    require(lenBytes.length == 8, "truncated safetensors length")
    val headerLen = ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).getLong()
    require(headerLen > 0 && headerLen <= MaxHeader && headerLen <= Int.MaxValue, "invalid or oversized safetensors header")
    val rawHeader = in.readNBytes(headerLen.toInt)
    require(rawHeader.length == headerLen.toInt, "truncated safetensors header")
    val headerObj = NativeJson.parse(new String(rawHeader, UTF_8)) match
      case Obj(fields) => fields
      case _ => throw IllegalArgumentException("safetensors header must be an object")
    headerObj.get("__metadata__") match
      case None => ()
      case Some(Obj(meta)) if meta.values.forall(_.isInstanceOf[Str]) => ()
      case _ => throw IllegalArgumentException("safetensors metadata must be a string map")
    val intervals = Vector.newBuilder[(Long, Long)]
    for (name, descJson) <- headerObj if name != "__metadata__" do
      val desc = descJson match
        case Obj(fields) => fields
        case _ => throw IllegalArgumentException("invalid tensor descriptor")
      val dtype = desc.get("dtype") match
        case Some(Str(dt)) if Bits.contains(dt) => dt
        case _ => throw IllegalArgumentException("invalid tensor shape, offsets or dtype")
      val shape = desc.get("shape") match
        case Some(Arr(dims)) => dims.map {
          case Number(n) if n.isValidLong && n >= 0 => n.toLong
          case _ => throw IllegalArgumentException("invalid tensor shape, offsets or dtype")
        }
        case _ => throw IllegalArgumentException("invalid tensor shape, offsets or dtype")
      val (start, end) = desc.get("data_offsets") match
        case Some(Arr(Vector(Number(s), Number(e)))) if s.isValidLong && e.isValidLong && s >= 0 && e >= 0 =>
          (s.toLong, e.toLong)
        case _ => throw IllegalArgumentException("invalid tensor shape, offsets or dtype")
      val elements = shape.foldLeft(1L)((acc, d) => Math.multiplyExact(acc, d))
      val bits = Math.multiplyExact(elements, Bits(dtype).toLong)
      require(bits % 8L == 0L && (end - start) == bits / 8L, "tensor size does not match descriptor")
      intervals += ((start, end))
    var expectedStart = 0L
    for (start, stop) <- intervals.result().sortBy(_._1) do
      require(start == expectedStart, "tensor data has overlaps or holes")
      expectedStart = stop
    require(fileSize == 8L + headerLen + expectedStart, "truncated or trailing tensor data")
    (headerObj, expectedStart, rawHeader)

  private def renderPretty(value: NativeJson, indent: Int = 0): String =
    val pad = " " * indent
    val innerPad = " " * (indent + 2)
    value match
      case Obj(fields) if fields.isEmpty => "{}"
      case Obj(fields) =>
        val entries = fields.toVector.sortBy(_._1).map { (k, v) =>
          s"$innerPad${NativeJson.render(Str(k))}: ${renderPretty(v, indent + 2)}"
        }
        s"{\n${entries.mkString(",\n")}\n$pad}"
      case Arr(values) if values.isEmpty => "[]"
      case Arr(values) =>
        val entries = values.map(v => s"$innerPad${renderPretty(v, indent + 2)}")
        s"[\n${entries.mkString(",\n")}\n$pad]"
      case other => NativeJson.render(other)

  private def parseExpectedToken(token: String): (FileIdentity, String) =
    require(token.nonEmpty && token.length <= 1024, "Load metadata first; a source token is required for saving.")
    NativeJson.parse(token) match
      case Arr(Vector(Arr(Vector(Number(_), Number(kh), Number(sz), Number(mt), Number(ct))), Str(hash)))
          if kh.isValidLong && sz.isValidLong && mt.isValidLong && ct.isValidLong && hash.matches("[0-9a-f]{64}") =>
        (FileIdentity(sz.toLong, mt.toLong, ct.toLong, kh.toLong), hash)
      case _ => throw IllegalArgumentException("Invalid source token; reload metadata before saving.")

  private def parseMetadataJson(text: String): Map[String, NativeJson] =
    require(text.getBytes(UTF_8).length <= MaxMetadata, "metadata exceeds 4 MiB")
    val obj = NativeJson.parse(text) match
      case Obj(fields) => fields
      case _ => throw IllegalArgumentException("metadata must be a JSON object")
    obj.map { (k, v) =>
      val strVal = v match
        case Str(s) => s
        case other => NativeJson.render(other)
      k -> Str(strVal)
    }

  private def resolveTarget(source: Path, newName: String): Path =
    val stem = if newName.nonEmpty then newName else
      val fn = source.getFileName.toString
      fn.substring(0, fn.length - ".safetensors".length) + "_md"
    require(
      stem.length <= 200 && stem != "." && stem != ".." && !stem.endsWith(" ") && !stem.endsWith(".") &&
        !stem.exists(c => c < 32 || "<>:\"/\\|?*".indexOf(c) >= 0) &&
        !stem.matches("(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\\..*|$)"),
      "output name must be a valid simple filename stem"
    )
    val target = source.resolveSibling(stem + ".safetensors")
    if Files.exists(target, java.nio.file.LinkOption.NOFOLLOW_LINKS) then
      throw IllegalArgumentException(s"Output already exists: $target")
    target

  def execute(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val allowed = Set("operation", "path", "metadata_json", "new_name", "source_token")
    require((params.keySet -- allowed).isEmpty, "unknown metadata_utils parameter")
    def str(key: String, default: String = ""): String = params.get(key) match
      case None | Some(Null) => default
      case Some(Str(s)) => s
      case _ => throw IllegalArgumentException(s"$key must be a string")
    context.checkCancelled()
    val operation = str("operation", "read")
    if operation == "read" then
      val path = resolveSource(str("path"))
      val initial = identityOf(path)
      val (headerObj, totalBytes, rawHeader) = Using.resource(Files.newInputStream(path)) { in =>
        val (hdr, total, raw) = parseHeader(in, initial.size)
        val digest = MessageDigest.getInstance("SHA-256")
        val lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(raw.length.toLong).array()
        digest.update(lenBuf)
        digest.update(raw)
        var done = 0L
        val buf = new Array[Byte](Chunk)
        ensureUnchanged(path, initial)
        while done < total do
          context.checkCancelled()
          ensureUnchanged(path, initial)
          val toRead = math.min(Chunk.toLong, total - done).toInt
          val n = in.read(buf, 0, toRead)
          require(n > 0, "Source changed; reload metadata before saving.")
          digest.update(buf, 0, n)
          done += n
        context.checkCancelled()
        ensureUnchanged(path, initial)
        (hdr, total, HexFormat.of().formatHex(digest.digest()))
      }
      val metadata = headerObj.get("__metadata__") match
        case Some(Obj(fields)) => fields
        case _ => Map.empty[String, NativeJson]
      val editable = metadata.map { (k, v) =>
        val s = v.asInstanceOf[Str].value
        val parsed = if s.startsWith("{") then
          try NativeJson.parse(s) catch case NonFatal(_) => v
        else v
        k -> parsed
      }
      val token = NativeJson.render(Arr(Vector(initial.toJson, Str(rawHeader))))
      Map(
        "path" -> Str(path.toString),
        "metadata" -> Obj(metadata),
        "editable" -> Str(renderPretty(Obj(editable))),
        "tensor_bytes" -> Number(BigDecimal(totalBytes)),
        "source_token" -> Str(token)
      )
    else if operation == "write" then
      val (expectedIdentity, expectedHash) = parseExpectedToken(str("source_token"))
      val updatedMetadata = parseMetadataJson(str("metadata_json"))
      val path = resolveSource(str("path"))
      val target = resolveTarget(path, str("new_name"))
      val temp = target.resolveSibling(s".metadata-${UUID.randomUUID()}.part")
      try
        val (totalBytes, tensorSha) = Using.resource(Files.newInputStream(path)) { source =>
          ensureUnchanged(path, expectedIdentity)
          val (headerObj, total, originalRaw) = parseHeader(source, expectedIdentity.size)
          val sourceDigest = MessageDigest.getInstance("SHA-256")
          sourceDigest.update(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(originalRaw.length.toLong).array())
          sourceDigest.update(originalRaw)
          val nextHeaderObj = headerObj.updated("__metadata__", Obj(updatedMetadata))
          val rawJsonBytes = NativeJson.render(Obj(nextHeaderObj)).getBytes(UTF_8)
          val padLen = (8 - (rawJsonBytes.length % 8)) % 8
          val paddedHeader = if padLen == 0 then rawJsonBytes else
            val arr = new Array[Byte](rawJsonBytes.length + padLen)
            System.arraycopy(rawJsonBytes, 0, arr, 0, rawJsonBytes.length)
            java.util.Arrays.fill(arr, rawJsonBytes.length, arr.length, ' '.toByte)
            arr
          require(paddedHeader.length <= MaxHeader, "updated header exceeds 100 MiB")
          val tensorDigest = MessageDigest.getInstance("SHA-256")
          Using.resource(Files.newOutputStream(temp)) { out =>
            out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(paddedHeader.length.toLong).array())
            out.write(paddedHeader)
            var done = 0L
            val buf = new Array[Byte](Chunk)
            while done < total do
              context.checkCancelled()
              ensureUnchanged(path, expectedIdentity)
              val toRead = math.min(Chunk.toLong, total - done).toInt
              val n = source.read(buf, 0, toRead)
              require(n > 0, "source changed during metadata copy")
              out.write(buf, 0, n)
              tensorDigest.update(buf, 0, n)
              sourceDigest.update(buf, 0, n)
              done += n
            context.checkCancelled()
            ensureUnchanged(path, expectedIdentity)
            val actualSourceHash = HexFormat.of().formatHex(sourceDigest.digest())
            require(actualSourceHash == expectedHash, "Source changed; reload metadata before saving.")
            out.flush()
          }
          (total, HexFormat.of().formatHex(tensorDigest.digest()))
        }
        context.checkCancelled()
        ensureUnchanged(path, expectedIdentity)
        context.beginPublication()
        Files.createLink(target, temp)
        Map(
          "path" -> Str(target.toString),
          "tensor_bytes" -> Number(BigDecimal(totalBytes)),
          "tensor_sha256" -> Str(tensorSha)
        )
      finally Files.deleteIfExists(temp)
    else throw IllegalArgumentException("operation must be read or write")
