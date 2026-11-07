package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd, Variable}
import scalatorch.nn.LinearAdapterScope
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, StandardCopyOption}
import java.util.Locale
import scala.collection.mutable

// ============================================================================
// 1. Clip I/O (`clip_io`)
// ============================================================================

final case class ClipIoConfig(
    inputEnabled: Boolean = false,
    inputName: String = "",
    inputMode: String = "off",
    negativeInputName: String = "",
    negativeInputMode: String = "off",
    inputFormat: String = "csv",
    outputEnabled: Boolean = false,
    outputName: String = "",
    negativeOutputName: String = "",
    outputFormat: String = "csv",
    overwrite: Boolean = false,
    storageDir: String = ""
):
  require(ClipIo.Modes.contains(inputMode), s"invalid Clip I/O input mode '${inputMode}'")
  require(ClipIo.Modes.contains(negativeInputMode), s"invalid Clip I/O negative input mode '${negativeInputMode}'")
  require(ClipIo.Formats.contains(inputFormat), s"invalid Clip I/O input format '${inputFormat}'")
  require(ClipIo.Formats.contains(outputFormat), s"invalid Clip I/O output format '${outputFormat}'")

object ClipIo:
  val Modes: Set[String] = Set("off", "replace", "append")
  val Formats: Set[String] = Set("csv", "safetensors", "pt")
  private val SafeFilename = "^(?!\\.{1,2}$)[A-Za-z0-9._() +=,\\-]+$".r
  private val PooledMarker = ", #POOLED:"

  sealed trait Expr
  final case class NumExpr(value: Double) extends Expr
  final case class NameExpr(name: String) extends Expr
  final case class UnaryNeg(inner: Expr) extends Expr
  final case class BinOp(op: Char, left: Expr, right: Expr) extends Expr

  def resolveStorageDir(config: ClipIoConfig, env: Map[String, String] = sys.env): Path =
    val raw = if config.storageDir.trim.nonEmpty then config.storageDir.trim
      else env.getOrElse("SDFX_CLIP_IO_DIR", "").trim
    val base = if raw.nonEmpty then Path.of(raw)
      else Path.of(System.getProperty("user.home", ".")).resolve(".cache").resolve("stablediffusion-nui").resolve("clip_io")
    Files.createDirectories(base)
    base.toRealPath()

  def safeTarget(base: Path, rawName: String, format: String): Path =
    val trimmed = rawName.trim
    require(trimmed.nonEmpty, "Clip I/O file name is empty")
    val ext = "." + format
    val filename = if trimmed.toLowerCase(Locale.ROOT).endsWith(ext) then trimmed else trimmed + ext
    require(SafeFilename.matches(filename), s"invalid Clip I/O file name '${rawName}'")
    val candidate = base.resolve(filename).normalize()
    require(candidate.getParent == base, s"Clip I/O path escapes storage directory: '${rawName}'")
    require(!Files.exists(candidate, java.nio.file.LinkOption.NOFOLLOW_LINKS) ||
      Files.isRegularFile(candidate, java.nio.file.LinkOption.NOFOLLOW_LINKS),
      s"Clip I/O target '${filename}' is not a regular file")
    candidate

  def parseExpression(spec: String): Expr =
    val tokens = tokenizeExpr(spec)
    require(tokens.nonEmpty, "Clip Input expression is empty")
    var pos = 0
    def peek: Option[String] = if pos < tokens.size then Some(tokens(pos)) else None
    def next(): String = { val t = tokens(pos); pos += 1; t }

    def parseAddSub(): Expr =
      var left = parseMulDiv()
      while peek.exists(t => t == "+" || t == "-") do
        val op = next().head
        val right = parseMulDiv()
        left = BinOp(op, left, right)
      left

    def parseMulDiv(): Expr =
      var left = parseUnary()
      while peek.exists(t => t == "*" || t == "/") do
        val op = next().head
        val right = parseUnary()
        left = BinOp(op, left, right)
      left

    def parseUnary(): Expr =
      if peek.contains("+") then { next(); parseUnary() }
      else if peek.contains("-") then { next(); UnaryNeg(parseUnary()) }
      else if peek.contains("(") then
        next()
        val inner = parseAddSub()
        require(peek.contains(")"), "unbalanced parentheses in Clip Input expression")
        next()
        inner
      else
        require(pos < tokens.size, "unexpected end of Clip Input expression")
        val tok = next()
        require(tok != ")", "unexpected ')' in Clip Input expression")
        tok.toDoubleOption match
          case Some(d) if d.isFinite => NumExpr(d)
          case _ => NameExpr(tok)

    val result = parseAddSub()
    require(pos == tokens.size, s"unexpected token '${tokens(pos)}' in Clip Input expression")
    result

  private def tokenizeExpr(spec: String): Vector[String] =
    val s = spec.trim
    if s.isEmpty then return Vector.empty
    if SafeFilename.matches(s) && !s.exists(ch => ch == '+' || ch == '*' || ch == '/' || ch == '(' || ch == ')') then
      return Vector(s)
    val out = mutable.ArrayBuffer.empty[String]
    val buf = new StringBuilder
    def flush(): Unit =
      val t = buf.toString.trim
      buf.clear()
      if t.nonEmpty then out += t
    var i = 0
    while i < s.length do
      val ch = s.charAt(i)
      if ch == '+' || ch == '*' || ch == '/' || ch == '(' || ch == ')' then
        flush()
        out += ch.toString
      else if ch == '-' then
        val prev = buf.toString.trim
        val prevTok = if prev.nonEmpty then prev else out.lastOption.getOrElse("")
        val isOperator = prevTok.isEmpty || Set("+", "-", "*", "/", "(").contains(prevTok) ||
          prev.isEmpty && i > 0 && s.charAt(i - 1).isWhitespace ||
          (i + 1 < s.length && s.charAt(i + 1).isWhitespace)
        if isOperator then
          flush()
          out += "-"
        else
          buf.append(ch)
      else
        buf.append(ch)
      i += 1
    flush()
    out.toVector

  final case class CondData(cond: Tensor[Float], pooled: Option[Tensor[Float]])

  def writeConditioning(target: Path, format: String, data: CondData): Unit =
    val cond2d = if data.cond.shape.rank == 3 then
      require(data.cond.shape.dims.head == 1L, "Clip Output expects batch size 1 conditioning")
      data.cond.reshape(Shape.from(Seq(data.cond.shape.dims(1), data.cond.shape.dims(2))))
    else data.cond
    val rows = cond2d.shape.dims(0).toInt
    val cols = cond2d.shape.dims(1).toInt
    val arr = cond2d.toFloatArray
    val tmp = Files.createTempFile(target.getParent, ".clip_io_", ".tmp")
    try
      format match
        case "csv" =>
          val sb = new java.lang.StringBuilder(rows * cols * 10)
          for r <- 0 until rows do
            for c <- 0 until cols do
              if c > 0 then sb.append(", ")
              sb.append(java.lang.Float.toString(arr(r * cols + c)))
            sb.append('\n')
          data.pooled.foreach { p =>
            val pArr = p.toFloatArray
            sb.append(PooledMarker).append('\n')
            for c <- 0 until pArr.length do
              if c > 0 then sb.append(", ")
              sb.append(java.lang.Float.toString(pArr(c)))
            sb.append('\n')
          }
          Files.writeString(tmp, sb.toString, StandardCharsets.UTF_8)
        case "safetensors" | "pt" =>
          val entries = mutable.ArrayBuffer[(String, Tensor[Float])]("cond" -> cond2d)
          data.pooled.foreach { p =>
            val p1d = p.reshape(Shape.from(Seq(p.shape.numel)))
            entries += ("pooled" -> p1d)
          }
          writeSafetensorsFile(tmp, entries.toVector)
      Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING)
    finally
      Files.deleteIfExists(tmp)

  def readConditioning(target: Path, format: String): CondData =
    require(Files.isRegularFile(target), s"Clip Input file not found: ${target.getFileName}")
    format match
      case "csv" =>
        val lines = Files.readAllLines(target, StandardCharsets.UTF_8)
        val condRows = mutable.ArrayBuffer.empty[Array[Float]]
        val pooledRows = mutable.ArrayBuffer.empty[Array[Float]]
        var inPooled = false
        import scala.jdk.CollectionConverters.*
        for rawLine <- lines.asScala do
          val s = rawLine.trim
          if s.isEmpty then ()
          else if s == PooledMarker then inPooled = true
          else if s.startsWith(",") || s.startsWith("#") then ()
          else
            val vals = s.split(",").map { cell =>
              val v = cell.trim.toFloatOption.getOrElse(throw IllegalArgumentException(s"non-numeric value in ${target.getFileName}"))
              require(v.isFinite, s"non-finite value in ${target.getFileName}")
              v
            }
            if inPooled then pooledRows += vals else condRows += vals
        require(condRows.nonEmpty && condRows.head.nonEmpty, s"Clip Input CSV '${target.getFileName}' contains no conditioning rows")
        val cols = condRows.head.length
        require(condRows.forall(_.length == cols), s"inconsistent column count in '${target.getFileName}'")
        val flat = new Array[Float](condRows.size * cols)
        for r <- condRows.indices do Array.copy(condRows(r), 0, flat, r * cols, cols)
        val condTensor = Tensor.fromFloats(flat, Shape.from(Seq(condRows.size.toLong, cols.toLong)))
        val pooledOpt = if pooledRows.nonEmpty then
          require(pooledRows.size == 1, s"pooled section in '${target.getFileName}' must have 1 row")
          Some(Tensor.fromFloats(pooledRows.head, Shape.from(Seq(1L, pooledRows.head.length.toLong))))
        else None
        CondData(condTensor, pooledOpt)
      case "safetensors" | "pt" =>
        scala.util.Using.resource(scalatorch.io.SafeTensors.open(target)) { archive =>
          val condKey = if archive.tensors.contains("cond") then "cond"
            else if archive.tensors.size == 1 then archive.tensors.keys.head
            else throw IllegalArgumentException(s"${target.getFileName} has no 'cond' tensor")
          val rawCond = archive.readFloat(condKey)
          val cond2d = if rawCond.shape.rank == 3 && rawCond.shape.dims.head == 1L then
            rawCond.reshape(Shape.from(Seq(rawCond.shape.dims(1), rawCond.shape.dims(2))))
          else if rawCond.shape.rank == 2 then rawCond
          else throw IllegalArgumentException(s"${target.getFileName} conditioning tensor must be 2D")
          require(cond2d.toFloatArray.forall(_.isFinite), s"${target.getFileName} contains non-finite values")
          val pooledOpt = if archive.tensors.contains("pooled") then
            val p = archive.readFloat("pooled")
            require(p.toFloatArray.forall(_.isFinite), s"${target.getFileName} pooled tensor contains non-finite values")
            Some(p.reshape(Shape.from(Seq(1L, p.shape.numel))))
          else None
          CondData(cond2d, pooledOpt)
        }

  def evalExpr(expr: Expr, load: String => CondData): Either[Double, CondData] = expr match
    case NumExpr(v) => Left(v)
    case NameExpr(name) => Right(load(name))
    case UnaryNeg(inner) =>
      evalExpr(inner, load) match
        case Left(v) => Left(-v)
        case Right(cd) =>
          Right(CondData(
            Tensor.fromFloats(cd.cond.toFloatArray.map(x => -x), cd.cond.shape),
            cd.pooled.map(p => Tensor.fromFloats(p.toFloatArray.map(x => -x), p.shape))
          ))
    case BinOp(op, lExpr, rExpr) =>
      val l = evalExpr(lExpr, load)
      val r = evalExpr(rExpr, load)
      (op, l, r) match
        case ('+', Left(a), Left(b)) => Left(a + b)
        case ('-', Left(a), Left(b)) => Left(a - b)
        case ('*', Left(a), Left(b)) => Left(a * b)
        case ('/', Left(a), Left(b)) =>
          require(b != 0.0, "division by zero in Clip Input expression")
          Left(a / b)
        case ('*', Left(s), Right(cd)) => Right(scaleCond(cd, s.toFloat))
        case ('*', Right(cd), Left(s)) => Right(scaleCond(cd, s.toFloat))
        case ('/', Right(cd), Left(s)) =>
          require(s != 0.0, "division by zero in Clip Input expression")
          Right(scaleCond(cd, (1.0 / s).toFloat))
        case ('+' | '-', Right(a), Right(b)) =>
          require(a.cond.shape.dims(1) == b.cond.shape.dims(1),
            s"Clip Input embedding dimension mismatch (${a.cond.shape.dims(1)} vs ${b.cond.shape.dims(1)})")
          val sign = if op == '+' then 1f else -1f
          val cols = a.cond.shape.dims(1).toInt
          val rowsA = a.cond.shape.dims(0).toInt
          val rowsB = b.cond.shape.dims(0).toInt
          val maxRows = math.max(rowsA, rowsB)
          val arrA = padRows(a.cond.toFloatArray, rowsA, maxRows, cols)
          val arrB = padRows(b.cond.toFloatArray, rowsB, maxRows, cols)
          val out = Array.tabulate(maxRows * cols) { i =>
            val v = arrA(i) + sign * arrB(i)
            require(v.isFinite, "Clip Input expression produced non-finite values")
            v
          }
          val pooledOut = (a.pooled, b.pooled) match
            case (Some(pa), Some(pb)) =>
              require(pa.shape == pb.shape, "Clip Input pooled dimension mismatch")
              val paArr = pa.toFloatArray; val pbArr = pb.toFloatArray
              Some(Tensor.fromFloats(Array.tabulate(paArr.length)(i => paArr(i) + sign * pbArr(i)), pa.shape))
            case (Some(pa), None) => Some(pa)
            case (None, Some(pb)) => Some(Tensor.fromFloats(pb.toFloatArray.map(_ * sign), pb.shape))
            case (None, None) => None
          Right(CondData(Tensor.fromFloats(out, Shape.from(Seq(maxRows.toLong, cols.toLong))), pooledOut))
        case _ =>
          throw IllegalArgumentException(s"cannot apply '${op}' between scalar and conditioning in Clip Input expression")

  private def scaleCond(cd: CondData, factor: Float): CondData =
    val c = Tensor.fromFloats(cd.cond.toFloatArray.map { x =>
      val v = x * factor
      require(v.isFinite, "Clip Input expression produced non-finite values")
      v
    }, cd.cond.shape)
    val p = cd.pooled.map(t => Tensor.fromFloats(t.toFloatArray.map(_ * factor), t.shape))
    CondData(c, p)

  private def padRows(arr: Array[Float], rows: Int, targetRows: Int, cols: Int): Array[Float] =
    if rows == targetRows then arr
    else
      val out = new Array[Float](targetRows * cols)
      Array.copy(arr, 0, out, 0, rows * cols)
      val lastRowOffset = (rows - 1) * cols
      for r <- rows until targetRows do
        Array.copy(arr, lastRowOffset, out, r * cols, cols)
      out

  def applySide(
      mode: String,
      spec: String,
      baseCond: Tensor[Float],
      basePooled: Option[Tensor[Float]],
      emptyChunk: Tensor[Float],
      storageDir: Path,
      format: String
  ): (Tensor[Float], Option[Tensor[Float]]) =
    if mode == "off" then return (baseCond, basePooled)
    val loaded = evalExpr(parseExpression(spec), name => readConditioning(safeTarget(storageDir, name, format), format)) match
      case Right(cd) => cd
      case Left(_) => throw IllegalArgumentException("Clip Input expression evaluated to a scalar instead of conditioning")
    val rows = loaded.cond.shape.dims(0).toInt
    val cols = loaded.cond.shape.dims(1).toInt
    val expectedCols = baseCond.shape.dims(2).toInt
    require(cols == expectedCols, s"Clip Input conditioning width ${cols} does not match model text width ${expectedCols}")
    require(rows > 0 && rows % 77 == 0, s"Clip Input conditioning row count ${rows} must be a positive multiple of 77")
    val cond3d = loaded.cond.reshape(Shape.from(Seq(1L, rows.toLong, cols.toLong)))
    mode match
      case "replace" =>
        val nextPooled = basePooled.map { bp =>
          loaded.pooled.filter(_.shape == bp.shape).getOrElse(bp)
        }
        (cond3d, nextPooled)
      case "append" =>
        val bArr = baseCond.toFloatArray
        val lArr = cond3d.toFloatArray
        val joined = new Array[Float](bArr.length + lArr.length)
        Array.copy(bArr, 0, joined, 0, bArr.length)
        Array.copy(lArr, 0, joined, bArr.length, lArr.length)
        val totalRows = baseCond.shape.dims(1) + rows.toLong
        (Tensor.fromFloats(joined, Shape.from(Seq(1L, totalRows, cols.toLong))), basePooled)
      case other =>
        throw IllegalArgumentException(s"unsupported Clip Input mode '${other}'")

  private def writeSafetensorsFile(path: Path, tensors: Vector[(String, Tensor[Float])]): Unit =
    var offset = 0L
    val headerFields = mutable.ArrayBuffer.empty[String]
    val payloads = tensors.map { case (name, t) =>
      val floats = t.toFloatArray
      val byteLen = floats.length.toLong * 4L
      val shapeStr = t.shape.dims.mkString("[", ",", "]")
      val start = offset
      val end = offset + byteLen
      offset = end
      headerFields += s"\"${name}\":{\"dtype\":\"F32\",\"shape\":${shapeStr},\"data_offsets\":[${start},${end}]}"
      floats
    }
    val jsonBytes = ("{" + headerFields.mkString(",") + "}").getBytes(StandardCharsets.UTF_8)
    val pad = (8 - (jsonBytes.length % 8)) % 8
    val paddedLen = jsonBytes.length + pad
    val out = Files.newOutputStream(path)
    try
      val lenBuf = java.nio.ByteBuffer.allocate(8).order(java.nio.ByteOrder.LITTLE_ENDIAN).putLong(paddedLen.toLong)
      out.write(lenBuf.array())
      out.write(jsonBytes)
      if pad > 0 then out.write(Array.fill[Byte](pad)(' '.toByte))
      payloads.foreach { floats =>
        val bb = java.nio.ByteBuffer.allocate(floats.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(floats)
        out.write(bb.array())
      }
    finally out.close()

// ============================================================================
// 2. Prompt Fusion (`prompt_fusion`)
// ============================================================================

object PromptFusion:
  private val VarDeclPattern = "(?m)^\\s*\\$([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*(.+)$".r
  private val VarUsePattern = "\\$([A-Za-z_][A-Za-z0-9_]*)".r
  private val InterpFunctions = Set("linear", "bezier", "catmull", "mean", "slerp")

  final case class FusionSpec(
      controlPrompts: Vector[String],
      startRatio: Double,
      endRatio: Double,
      func: String
  )

  def applyDeclarations(text: String): String =
    if !text.contains("$") then return text
    val vars = mutable.LinkedHashMap.empty[String, String]
    val remainingLines = mutable.ArrayBuffer.empty[String]
    for line <- text.linesIterator do
      line match
        case VarDeclPattern(name, value) =>
          val resolvedVal = VarUsePattern.replaceAllIn(value.trim, m => java.util.regex.Matcher.quoteReplacement(vars.getOrElse(m.group(1), m.matched)))
          vars(name) = resolvedVal
        case _ =>
          remainingLines += line
    if vars.isEmpty then text
    else
      val body = remainingLines.mkString("\n")
      VarUsePattern.replaceAllIn(body, m => java.util.regex.Matcher.quoteReplacement(vars.getOrElse(m.group(1), m.matched))).trim

  /** Detects whether `text` uses Prompt Fusion syntax (`$var = ...`, `[a:b:...:linear|bezier|catmull|mean]`, or `(text:w1,w2)`). */
  def isActive(text: String): Boolean =
    if text == null || text.isEmpty then false
    else
      val withDecls = applyDeclarations(text)
      if withDecls != text then true
      else parseSpec(withDecls, 10).isDefined

  /** Parses top-level and embedded Prompt Fusion constructs into a control-point specification. */
  def parseSpec(rawText: String, totalSteps: Int): Option[FusionSpec] =
    val text = applyDeclarations(rawText)
    // 1. Check weight interpolation: (word:w1,w2) or (word:w1,w2:start,end)
    val weightInterp = "\\(([^():]+):\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*,\\s*([+-]?\\d+(?:\\.\\d+)?)(?:\\s*:\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*,\\s*([+-]?\\d+(?:\\.\\d+)?))?\\)".r
    weightInterp.findFirstMatchIn(text) match
      case Some(m) =>
        val inner = m.group(1)
        val w1 = m.group(2)
        val w2 = m.group(3)
        val sRatio = Option(m.group(4)).flatMap(_.toDoubleOption).map(v => normalizeBound(v, totalSteps)).getOrElse(0.0)
        val eRatio = Option(m.group(5)).flatMap(_.toDoubleOption).map(v => normalizeBound(v, totalSteps)).getOrElse(1.0)
        require(sRatio <= eRatio, "Prompt Fusion start step must not exceed end step")
        val p1 = text.substring(0, m.start) + s"(${inner}:${w1})" + text.substring(m.end)
        val p2 = text.substring(0, m.start) + s"(${inner}:${w2})" + text.substring(m.end)
        return Some(FusionSpec(Vector(p1, p2), sRatio, eRatio, "linear"))
      case None => ()

    // 2. Check bracketed functional interpolation: [p1 : p2 (: p3 ...) : bounds : func] or [p1 : p2 : start, end]
    findTopBracket(text) match
      case Some((bStart, bEnd)) =>
        val inside = text.substring(bStart + 1, bEnd - 1)
        val parts = splitTopColons(inside)
        if parts.size >= 3 then
          val last = parts.last.trim.toLowerCase(Locale.ROOT)
          if InterpFunctions.contains(last) && parts.size >= 4 then
            val boundsStr = parts(parts.size - 2).trim
            val (sRatio, eRatio) = parseBounds(boundsStr, totalSteps)
            val prompts = parts.dropRight(2).map(p => text.substring(0, bStart) + p + text.substring(bEnd))
            return Some(FusionSpec(prompts, sRatio, eRatio, last))
          else if parts.last.contains(",") then
            parseBoundsOpt(parts.last.trim, totalSteps) match
              case Some((sRatio, eRatio)) =>
                val prompts = parts.dropRight(1).map(p => text.substring(0, bStart) + p + text.substring(bEnd))
                return Some(FusionSpec(prompts, sRatio, eRatio, "linear"))
              case None => ()
        None
      case None => None

  private def normalizeBound(v: Double, totalSteps: Int): Double =
    val r = if v > 1.0 then (v - 1.0) / math.max(1, totalSteps - 1).toDouble else v
    math.max(0.0, math.min(1.0, r))

  private def parseBoundsOpt(s: String, totalSteps: Int): Option[(Double, Double)] =
    val sp = s.split(",").map(_.trim)
    if sp.length == 2 then
      for
        a <- sp(0).toDoubleOption
        b <- sp(1).toDoubleOption
      yield
        val ra = normalizeBound(a, totalSteps)
        val rb = normalizeBound(b, totalSteps)
        require(ra <= rb, "Prompt Fusion start step must not exceed end step")
        (ra, rb)
    else if sp.length == 1 then
      sp(0).toDoubleOption.map(v => (0.0, normalizeBound(v, totalSteps)))
    else None

  private def parseBounds(s: String, totalSteps: Int): (Double, Double) =
    parseBoundsOpt(s, totalSteps).getOrElse(throw IllegalArgumentException(s"invalid Prompt Fusion step bounds '${s}'"))

  private def findTopBracket(s: String): Option[(Int, Int)] =
    var depth = 0
    var start = -1
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      if c == '\\' then i += 1
      else if c == '[' then
        if depth == 0 then start = i
        depth += 1
      else if c == ']' && depth > 0 then
        depth -= 1
        if depth == 0 && start >= 0 then return Some((start, i + 1))
      i += 1
    None

  private def splitTopColons(s: String): Vector[String] =
    val out = mutable.ArrayBuffer.empty[String]
    val cur = new StringBuilder
    var depth = 0
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      if c == '\\' && i + 1 < s.length then
        cur.append(c).append(s.charAt(i + 1))
        i += 1
      else if c == '[' || c == '(' then
        depth += 1
        cur.append(c)
      else if (c == ']' || c == ')') && depth > 0 then
        depth -= 1
        cur.append(c)
      else if c == ':' && depth == 0 then
        out += cur.toString
        cur.clear()
      else cur.append(c)
      i += 1
    out += cur.toString
    out.toVector

  /** Interpolate control-point tensors at step `stepIdx` (0-based) out of `totalSteps`. */
  def interpolateTensor(controls: Vector[Tensor[Float]], stepIdx: Int, totalSteps: Int, spec: FusionSpec): Tensor[Float] =
    require(controls.nonEmpty, "Prompt Fusion requires at least one control tensor")
    if controls.size == 1 then return controls.head
    val rawT = if totalSteps <= 1 then 0.0 else stepIdx.toDouble / (totalSteps - 1).toDouble
    if rawT <= spec.startRatio then return controls.head
    if rawT >= spec.endRatio then return controls.last
    val span = math.max(1e-9, spec.endRatio - spec.startRatio)
    val t = math.max(0.0, math.min(1.0, (rawT - spec.startRatio) / span))
    val shape = controls.head.shape
    val arrays = controls.map(_.toFloatArray)
    val len = arrays.head.length

    val out = spec.func match
      case "mean" =>
        val inv = (1.0 / arrays.size).toFloat
        Array.tabulate(len) { idx =>
          var sum = 0f
          var k = 0
          while k < arrays.size do { sum += arrays(k)(idx); k += 1 }
          sum * inv
        }
      case "bezier" =>
        val n = arrays.size - 1
        val weights = Array.tabulate(arrays.size) { i =>
          binomial(n, i) * math.pow(1.0 - t, n - i) * math.pow(t, i)
        }
        Array.tabulate(len) { idx =>
          var sum = 0.0
          var k = 0
          while k < arrays.size do { sum += weights(k) * arrays(k)(idx); k += 1 }
          sum.toFloat
        }
      case "catmull" =>
        val segs = arrays.size - 1
        val pos = t * segs
        val seg = math.min(segs - 1, math.max(0, pos.toInt))
        val localT = pos - seg
        val p0 = arrays(math.max(0, seg - 1))
        val p1 = arrays(seg)
        val p2 = arrays(math.min(segs, seg + 1))
        val p3 = arrays(math.min(segs, seg + 2))
        val t2 = localT * localT
        val t3 = t2 * localT
        Array.tabulate(len) { idx =>
          val v0 = p0(idx).toDouble; val v1 = p1(idx).toDouble; val v2 = p2(idx).toDouble; val v3 = p3(idx).toDouble
          (0.5 * ((2.0 * v1) + (-v0 + v2) * localT + (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3) * t2 + (-v0 + 3.0 * v1 - 3.0 * v2 + v3) * t3)).toFloat
        }
      case "slerp" =>
        val segs = arrays.size - 1
        val pos = t * segs
        val seg = math.min(segs - 1, math.max(0, pos.toInt))
        val localT = pos - seg
        slerpArrays(arrays(seg), arrays(seg + 1), localT)
      case _ => // "linear"
        val segs = arrays.size - 1
        val pos = t * segs
        val seg = math.min(segs - 1, math.max(0, pos.toInt))
        val localT = (pos - seg).toFloat
        val a = arrays(seg)
        val b = arrays(seg + 1)
        Array.tabulate(len)(idx => a(idx) * (1f - localT) + b(idx) * localT)

    Tensor.fromFloats(out, shape)

  private def binomial(n: Int, k: Int): Double =
    if k < 0 || k > n then 0.0
    else
      var res = 1.0
      for i <- 1 to math.min(k, n - k) do
        res = res * (n - i + 1) / i
      res

  def slerpArrays(a: Array[Float], b: Array[Float], t: Double): Array[Float] =
    var normA = 0.0; var normB = 0.0; var dot = 0.0
    var i = 0
    while i < a.length do
      val va = a(i).toDouble; val vb = b(i).toDouble
      normA += va * va; normB += vb * vb; dot += va * vb
      i += 1
    val denom = math.sqrt(normA) * math.sqrt(normB)
    if denom <= 1e-12 then
      val tf = t.toFloat
      return Array.tabulate(a.length)(idx => a(idx) * (1f - tf) + b(idx) * tf)
    val cosOmega = math.max(-1.0, math.min(1.0, dot / denom))
    val omega = math.acos(cosOmega)
    val sinOmega = math.sin(omega)
    if math.abs(sinOmega) < 1e-6 then
      val tf = t.toFloat
      Array.tabulate(a.length)(idx => a(idx) * (1f - tf) + b(idx) * tf)
    else
      val wa = (math.sin((1.0 - t) * omega) / sinOmega).toFloat
      val wb = (math.sin(t * omega) / sinOmega).toFloat
      Array.tabulate(a.length)(idx => a(idx) * wa + b(idx) * wb)

// ============================================================================
// 3. Neutral Prompt (`neutral_prompt`)
// ============================================================================

final case class NeutralPromptConfig(cfgRescale: Double = 0.0):
  require(cfgRescale.isFinite && cfgRescale >= 0.0 && cfgRescale <= 1.0,
    "Neutral Prompt CFG rescale must be between 0 and 1")

object NeutralPrompt:
  val Keywords: Vector[String] = Vector("AND_PERP", "AND_SALT", "AND_TOPK", "AND")
  private val KeywordRegex = "\\b(AND_PERP|AND_SALT|AND_TOPK|AND)\\b".r

  sealed trait Node:
    def kind: String
    def weight: Double

  final case class Leaf(text: String, weight: Double = 1.0, kind: String = "AND") extends Node
  final case class Group(children: Vector[Node], weight: Double = 1.0, kind: String = "AND") extends Node

  final case class BranchConditioning(
      cond: Tensor[Float],
      posAdded: Option[DiffusionAddedConditioning],
      posSigns: Option[Array[Float]] = None,
      loraScope: Map[scalatorch.nn.LowRankAdapter, Float] = Map.empty
  )

  final case class StepPlan(
      tree: Group,
      branches: Vector[BranchConditioning],
      phi: Double
  )

  def parse(prompt: String): Group =
    val parts = splitTopKeywords(prompt)
    val children = parts.map { case (kind, rawSegment) =>
      parseSegment(rawSegment, kind)
    }
    Group(children, 1.0, "AND")

  def leaves(group: Group): Vector[Leaf] =
    val buf = mutable.ArrayBuffer.empty[Leaf]
    def visit(n: Node): Unit = n match
      case l: Leaf => buf += l
      case g: Group => g.children.foreach(visit)
    visit(group)
    buf.toVector

  def isActive(prompt: String, cfgRescale: Double): Boolean =
    val tree = parse(prompt)
    val lvs = leaves(tree)
    cfgRescale != 0.0 || lvs.size != 1 || lvs.exists(n => n.weight != 1.0 || n.kind != "AND") || lvs.head.text != prompt

  private def splitTopKeywords(text: String): Vector[(String, String)] =
    val matches = KeywordRegex.findAllMatchIn(text).filter { m =>
      isTopLevel(text, m.start)
    }.toVector
    if matches.isEmpty then Vector(("AND", text))
    else
      val out = mutable.ArrayBuffer.empty[(String, String)]
      var curKind = "AND"
      var curStart = 0
      for m <- matches do
        out += ((curKind, text.substring(curStart, m.start)))
        curKind = m.group(1)
        curStart = m.end
      out += ((curKind, text.substring(curStart)))
      out.toVector

  private def isTopLevel(text: String, pos: Int): Boolean =
    var depth = 0
    var i = 0
    while i < pos do
      val c = text.charAt(i)
      if c == '\\' then i += 1
      else if c == '[' || c == '(' || c == '<' then depth += 1
      else if (c == ']' || c == ')' || c == '>') && depth > 0 then depth -= 1
      i += 1
    depth == 0

  private def parseSegment(raw: String, kind: String): Node =
    val trimmed = raw.trim
    val (body, weight) = splitTrailingWeight(trimmed)
    val bTrim = body.trim
    if bTrim.startsWith("[") && bTrim.endsWith("]") && isMatchingOuterBracket(bTrim) then
      val inner = bTrim.substring(1, bTrim.length - 1)
      val subParts = splitTopKeywords(inner)
      if subParts.size > 1 || splitTrailingWeight(inner.trim)._2 != 1.0 then
        val children = subParts.map { case (k, seg) => parseSegment(seg, k) }
        return Group(children, weight, kind)
    Leaf(if weight != 1.0 then body.trim else raw.trim, weight, kind)

  private def isMatchingOuterBracket(s: String): Boolean =
    var depth = 0
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      if c == '\\' then i += 1
      else if c == '[' then depth += 1
      else if c == ']' then
        depth -= 1
        if depth == 0 && i < s.length - 1 then return false
      i += 1
    depth == 0

  private def splitTrailingWeight(s: String): (String, Double) =
    var depth = 0
    var lastColon = -1
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      if c == '\\' then i += 1
      else if c == '[' || c == '(' || c == '<' then depth += 1
      else if (c == ']' || c == ')' || c == '>') && depth > 0 then depth -= 1
      else if c == ':' && depth == 0 then lastColon = i
      i += 1
    if lastColon < 0 then (s, 1.0)
    else
      val rhs = s.substring(lastColon + 1).trim
      rhs.toDoubleOption match
        case Some(w) =>
          require(w.isFinite, "Neutral Prompt weight must be finite")
          (s.substring(0, lastColon), w)
        case None => (s, 1.0)

  def perpendicular(anchor: Array[Float], target: Array[Float]): Array[Float] =
    var normSq = 0.0
    var dot = 0.0
    var i = 0
    while i < anchor.length do
      val a = anchor(i).toDouble
      normSq += a * a
      dot += a * target(i).toDouble
      i += 1
    val norm = math.sqrt(normSq)
    if norm <= 1e-12 then target.clone()
    else
      val scale = (dot / (norm * norm)).toFloat
      Array.tabulate(target.length)(idx => target(idx) - scale * anchor(idx))

  def salientBlend(anchor: Array[Float], target: Array[Float], weight: Double): Array[Float] =
    val len = anchor.length
    def normalStd(arr: Array[Float]): Array[Double] =
      var sum = 0.0
      var i = 0
      while i < len do { sum += arr(i).toDouble; i += 1 }
      val mean = sum / math.max(1, len)
      var sq = 0.0
      i = 0
      while i < len do
        val d = arr(i).toDouble - mean
        sq += d * d
        i += 1
      val std = math.sqrt(sq / math.max(1, len - 1))
      Array.tabulate(len) { idx =>
        val z = if std > 1e-12 then math.abs((arr(idx).toDouble - mean) / std) else 0.0
        math.sqrt(5.0 * z)
      }
    val sA = normalStd(anchor)
    val sB = normalStd(target)
    Array.tabulate(len) { idx =>
      val maxS = math.max(sA(idx), sB(idx))
      val ea = math.exp(sA(idx) - maxS) * 1.0
      val eb = math.exp(sB(idx) - maxS) * math.abs(weight)
      val denom = ea + eb
      val wa = if denom > 0.0 then ea / denom else 0.5
      val wb = if denom > 0.0 then (eb / denom) * math.signum(weight) else 0.0
      (wa * anchor(idx).toDouble + wb * target(idx).toDouble).toFloat
    }

  def topk(anchor: Array[Float], target: Array[Float], weight: Double): Array[Float] =
    val len = target.length
    val absVals = Array.tabulate(len)(i => math.abs(target(i)))
    java.util.Arrays.sort(absVals)
    val qIdx = math.min(len - 1, math.max(0, math.round(0.95f * (len - 1)).toInt))
    val threshold = absVals(qIdx)
    val wf = weight.toFloat
    Array.tabulate(len) { idx =>
      if math.abs(target(idx)) >= threshold then anchor(idx) + wf * target(idx)
      else anchor(idx)
    }

  def combineDeltas(tree: Group, leafDeltas: Vector[Tensor[Float]]): Tensor[Float] =
    require(leafDeltas.nonEmpty, "NeutralPrompt requires at least one branch delta")
    val shape = leafDeltas.head.shape
    val len = shape.numel.toInt
    var cursor = 0

    def evalGroup(g: Group): Array[Float] =
      val acc = new Array[Float](len)
      for child <- g.children do
        val w = child.weight
        val raw = child match
          case _: Leaf =>
            val arr = leafDeltas(cursor).toFloatArray
            cursor += 1
            arr
          case sub: Group =>
            evalGroup(sub)
        if w != 0.0 then
          child.kind match
            case "AND" =>
              val wf = w.toFloat
              var i = 0
              while i < len do { acc(i) += wf * raw(i); i += 1 }
            case "AND_PERP" =>
              val p = perpendicular(acc, raw)
              val wf = w.toFloat
              var i = 0
              while i < len do { acc(i) += wf * p(i); i += 1 }
            case "AND_SALT" =>
              val blended = salientBlend(acc, raw, w)
              Array.copy(blended, 0, acc, 0, len)
            case "AND_TOPK" =>
              val tk = topk(acc, raw, w)
              Array.copy(tk, 0, acc, 0, len)
      acc

    val combined = evalGroup(tree)
    Tensor.fromFloats(combined, shape)

  def rescale(guided: Tensor[Float], cond: Tensor[Float], phi: Double): Tensor[Float] =
    if phi == 0.0 then return guided
    val gArr = guided.toFloatArray
    val cArr = cond.toFloatArray
    val len = gArr.length
    def stdOf(arr: Array[Float]): Double =
      var sum = 0.0
      var i = 0
      while i < len do { sum += arr(i).toDouble; i += 1 }
      val mean = sum / math.max(1, len)
      var sq = 0.0
      i = 0
      while i < len do
        val d = arr(i).toDouble - mean
        sq += d * d
        i += 1
      math.sqrt(sq / math.max(1, len - 1))
    val stdG = stdOf(gArr)
    if stdG <= 0.0 then return guided
    val stdC = stdOf(cArr)
    val factor = phi * (stdC / stdG) + (1.0 - phi)
    val ff = factor.toFloat
    Tensor.fromFloats(Array.tabulate(len)(i => gArr(i) * ff), guided.shape)

// ============================================================================
// 4. Composable LoRA (`composable_lora`)
// ============================================================================

final case class ComposableLoraConfig(
    enabled: Boolean = true,
    optUcTextModel: Boolean = false,
    optUcDiffusionModel: Boolean = false,
    optPlotLoraWeight: Boolean = false,
    optSingleNoUc: Boolean = false,
    hiresGlobalStep: Boolean = false,
    customTimestamp: String = ""
)

object ComposableLora:
  private val TagRegex = "<(lora|lyco):([^<>]+)>".r
  private val SuperCmdRegex = "#(increment|decrease|cmd\\([^)]*\\))".r
  private val StepWindowRegex = "\\[\\s*\\[\\s*([^:\\]]+):\\s*:([^\\]]+)\\]\\s*:\\s*([^\\]]+)\\]".r
  private val StartAtRegex = "\\[\\s*(<(?:lora|lyco):[^>]+>)\\s*:\\s*([0-9.]+)\\s*\\]".r

  final case class Controller(
      name: String,
      baseWeight: Double,
      start: Double = 0.0,
      end: Double = -1.0,
      mode: String = "const" // "const" | "increment" | "decrease" | "cmd"
  ):
    def weightAt(step: Int, totalSteps: Int): Double =
      if step < 0 then return baseWeight
      val effStart = if start > 0.0 && start < 1.0 then start * totalSteps else start
      val effEnd = if end < 0.0 then totalSteps.toDouble else if end < 1.0 then end * totalSteps else end
      if step < effStart || step > effEnd then 0.0
      else
        val progress = if effEnd > effStart then (step - effStart) / (effEnd - effStart) else 0.0
        mode match
          case "increment" => baseWeight * progress
          case "decrease" => baseWeight * (1.0 - progress)
          case _ => baseWeight

  def cleanPrompt(text: String): String =
    val noWin = StepWindowRegex.replaceAllIn(text, "")
    val noStart = StartAtRegex.replaceAllIn(noWin, "")
    val noCmd = SuperCmdRegex.replaceAllIn(noStart, "")
    TagRegex.replaceAllIn(noCmd, "").replaceAll("(?m)[ \\t]{2,}", " ").trim

  def parseBranchControllers(branchText: String): Vector[Controller] =
    val mode = SuperCmdRegex.findFirstMatchIn(branchText).map(_.group(1)) match
      case Some("increment") => "increment"
      case Some("decrease") => "decrease"
      case _ => "const"
    val out = mutable.ArrayBuffer.empty[Controller]
    var rem = branchText

    for m <- StepWindowRegex.findAllMatchIn(rem) do
      val inner = m.group(1)
      val endVal = m.group(2).trim.toDoubleOption.getOrElse(-1.0)
      val startVal = m.group(3).trim.toDoubleOption.getOrElse(0.0)
      for tm <- TagRegex.findAllMatchIn(inner) do
        val (name, w) = parseTagItems(tm.group(2))
        out += Controller(name, w, startVal, endVal, mode)
    rem = StepWindowRegex.replaceAllIn(rem, "")

    for m <- StartAtRegex.findAllMatchIn(rem) do
      val inner = m.group(1)
      val startVal = m.group(2).trim.toDoubleOption.getOrElse(0.0)
      for tm <- TagRegex.findAllMatchIn(inner) do
        val (name, w) = parseTagItems(tm.group(2))
        out += Controller(name, w, startVal, -1.0, mode)
    rem = StartAtRegex.replaceAllIn(rem, "")

    for tm <- TagRegex.findAllMatchIn(rem) do
      val (name, w) = parseTagItems(tm.group(2))
      out += Controller(name, w, 0.0, -1.0, mode)

    out.toVector

  private def parseTagItems(body: String): (String, Double) =
    val parts = body.split(":").map(_.trim)
    val name = parts.headOption.getOrElse("")
    val weight = if parts.length > 1 then
      parts(1).toDoubleOption.getOrElse(throw IllegalArgumentException(s"invalid Composable LoRA weight in '<lora:${body}>'"))
    else 1.0
    require(weight.isFinite, "Composable LoRA weights must be finite")
    (name, weight)

  def evaluateBranchMultipliers(controllers: Vector[Controller], step: Int, totalSteps: Int): Map[String, Float] =
    val result = mutable.HashMap.empty[String, Float]
    for c <- controllers do
      val w = c.weightAt(step, totalSteps)
      val prev = result.getOrElse(c.name, 0f)
      if math.abs(w) >= math.abs(prev.toDouble) then
        result(c.name) = w.toFloat
    result.toMap

// ============================================================================
// 5. Embedding Merge (`embedding_merge`)
// ============================================================================

object EmbeddingMerge:
  val MaxExpression = 65536
  val MaxRows = 4096
  val MaxActions = 4096
  private val NumberRegex = "^([+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+))".r

  final case class Action(
      kind: String,
      textVal: String = "",
      numVal: Double = 0.0,
      intVal: Int = 0,
      whole: Boolean = false,
      stream: Option[String] = None
  )

  final case class Span(start: Int, end: Int, original: String, expression: String)

  def hasExpressions(prompt: String): Boolean =
    prompt != null && (prompt.contains("<'") || prompt.contains("{'"))

  def extractSpans(prompt: String): Vector[Span] =
    if !hasExpressions(prompt) then return Vector.empty
    require(prompt.length <= MaxExpression, "Embedding Merge: prompt input limit exceeded")
    val out = mutable.ArrayBuffer.empty[Span]
    var pos = 0
    while pos < prompt.length - 1 do
      val idx = findNextOpen(prompt, pos)
      if idx < 0 then pos = prompt.length
      else
        val span = readSpan(prompt, idx)
        out += span
        pos = span.end
    out.toVector

  private def findNextOpen(s: String, from: Int): Int =
    var i = from
    while i < s.length - 1 do
      val c = s.charAt(i)
      if (c == '<' || c == '{') && s.charAt(i + 1) == '\'' then return i
      i += 1
    -1

  private def readSpan(text: String, start: Int): Span =
    val close = if text.charAt(start) == '<' then '>' else '}'
    var quote = false
    var pos = start + 1
    while pos < text.length do
      val ch = text.charAt(pos)
      if ch == '\'' then
        if quote && pos + 1 < text.length && text.charAt(pos + 1) == '\'' then pos += 2
        else { quote = !quote; pos += 1 }
      else if ch == close && !quote then
        val orig = text.substring(start, pos + 1)
        val expr = text.substring(start + 1, pos)
        parseExpression(expr)
        return Span(start, pos + 1, orig, expr)
      else pos += 1
    throw IllegalArgumentException(s"Embedding Merge: unclosed merge expression at ${start}")

  def parseExpression(raw: String): Vector[Action] =
    require(raw != null && raw.length <= MaxExpression, "Embedding Merge: expression input limit exceeded")
    val text = raw.replace('\u0000', ' ')
    val actions = mutable.ArrayBuffer.empty[Action]
    var pos = 0
    def skipSpaces(): Unit =
      while pos < text.length && text.charAt(pos).isWhitespace do pos += 1
    def readQuoted(): String =
      skipSpaces()
      if pos >= text.length || text.charAt(pos) != '\'' then
        throw IllegalArgumentException(s"Embedding Merge: expected quoted text at ${pos}")
      pos += 1
      val sb = new StringBuilder
      while pos < text.length do
        val ch = text.charAt(pos)
        pos += 1
        if ch == '\'' then
          if pos < text.length && text.charAt(pos) == '\'' then { sb.append('\''); pos += 1 }
          else return sb.toString
        else sb.append(ch)
      throw IllegalArgumentException("Embedding Merge: last quote not closed")

    actions += Action("text", textVal = readQuoted())
    var done = false
    while !done do
      skipSpaces()
      if pos >= text.length then done = true
      else
        if actions.size >= MaxActions then throw IllegalArgumentException("Embedding Merge: expression action limit exceeded")
        val whole = text.charAt(pos) == '='
        if whole then { pos += 1; skipSpaces() }
        if pos >= text.length then throw IllegalArgumentException("Embedding Merge: unfinished result operator")
        val op = text.charAt(pos)
        pos += 1
        skipSpaces()
        if (op == '+' || op == '-') && !whole then
          actions += Action(if op == '+' then "add" else "sub")
          actions += Action("text", textVal = readQuoted())
        else if op == '*' || op == '/' then
          val sub = text.substring(pos)
          val m = NumberRegex.findFirstMatchIn(sub).getOrElse(
            throw IllegalArgumentException("Embedding Merge: expected numeric multiplier"))
          pos += m.end
          val v = m.group(1).toDouble
          skipSpaces()
          val stream = if pos < text.length && (text.charAt(pos) == 'L' || text.charAt(pos) == 'G') then
            val s = if text.charAt(pos) == 'L' then "clip_l" else "clip_g"
            pos += 1
            Some(s)
          else None
          actions += Action(if op == '*' then "mul" else "div", numVal = v, whole = whole, stream = stream)
        else if op == ':' then
          var shift = 0
          if pos < text.length && (text.charAt(pos) == '+' || text.charAt(pos) == '-') then
            shift = if text.charAt(pos) == '+' then 1 else -1
            pos += 1
            skipSpaces()
          val intPat = if shift != 0 then "^(-?[0-9]+)".r else "^([0-9]+)".r
          val m = intPat.findFirstMatchIn(text.substring(pos)).getOrElse(
            throw IllegalArgumentException("Embedding Merge: expected integer vector count or rotation"))
          pos += m.end
          val v = m.group(1).toInt
          if shift != 0 then actions += Action("roll", intVal = v * shift, whole = whole)
          else actions += Action("resize", intVal = v, whole = whole)
        else if op == ',' && !whole then
          if pos < text.length && text.charAt(pos) == '\'' then
            actions += Action("append", textVal = readQuoted())
          else
            val m = "^([0-9]+)".r.findFirstMatchIn(text.substring(pos)).getOrElse(
              throw IllegalArgumentException("Embedding Merge: expected quoted text or token ID after comma"))
            val rawTok = m.group(1)
            pos += rawTok.length
            if rawTok.startsWith("0") then
              rawTok match
                case "0" => actions += Action("token_id", intVal = 0)
                case "00" => actions += Action("token_special", textVal = "pad")
                case "000" => actions += Action("token_special", textVal = "end")
                case "0000" => actions += Action("token_special", textVal = "start")
                case _ => throw IllegalArgumentException("Embedding Merge: invalid zero-prefixed token ID")
            else
              actions += Action("token_id", intVal = rawTok.toInt)
        else
          throw IllegalArgumentException(s"Embedding Merge: invalid operator '${op}' at ${pos - 1}")
    actions.toVector

  /** StreamVectors: one `Vector[Array[Float]]` (rows x width) per stream (`clip_l`, optionally `clip_g`). */
  type StreamVectors = Vector[Vector[Array[Float]]]

  def evaluate(
      actions: Vector[Action],
      streamNames: Vector[String],
      streamWidths: Vector[Int],
      resolveText: String => StreamVectors,
      resolveToken: Either[Int, String] => StreamVectors
  ): StreamVectors =
    def checked(sv: StreamVectors): StreamVectors =
      require(sv.size == streamNames.size, "Embedding Merge: mismatched CLIP streams")
      for s <- sv.indices do
        val rows = sv(s)
        val w = streamWidths(s)
        require(rows.size <= MaxRows, "Embedding Merge: vector row limit exceeded")
        require(rows.forall(r => r.length == w && r.forall(_.isFinite)), "Embedding Merge: nonfinite or mismatched vector")
      require(sv.map(_.size).distinct.size <= 1, "Embedding Merge: CLIP stream rows must match")
      sv

    def resizeRows(rows: Vector[Array[Float]], count: Int, width: Int): Vector[Array[Float]] =
      require(count >= 0 && count <= MaxRows, "Embedding Merge: vector row limit exceeded")
      Vector.tabulate(count) { r =>
        if r < rows.size then rows(r).clone()
        else new Array[Float](width)
      }

    def rollRows(rows: Vector[Array[Float]], shift: Int): Vector[Array[Float]] =
      if rows.isEmpty then rows
      else
        val n = rows.size
        val k = ((shift % n) + n) % n
        if k == 0 then rows.map(_.clone())
        else (rows.drop(n - k) ++ rows.take(n - k)).map(_.clone())

    var left: Option[StreamVectors] = None
    var right: Option[StreamVectors] = None
    var sign = 1f

    def combine(): Unit =
      if left.isDefined then
        val l = left.get
        val r = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing right operand"))
        val count = math.max(l.head.size, r.head.size)
        val merged = Vector.tabulate(streamNames.size) { s =>
          val w = streamWidths(s)
          val la = resizeRows(l(s), count, w)
          val ra = resizeRows(r(s), count, w)
          Vector.tabulate(count) { rowIdx =>
            val rowL = la(rowIdx); val rowR = ra(rowIdx)
            Array.tabulate(w)(c => rowL(c) + sign * rowR(c))
          }
        }
        right = Some(checked(merged))
        left = None

    for action <- actions do
      if action.whole || action.kind == "add" || action.kind == "sub" then combine()
      action.kind match
        case "text" =>
          right = Some(checked(resolveText(action.textVal)))
        case "add" | "sub" =>
          left = right
          right = None
          sign = if action.kind == "add" then 1f else -1f
        case "append" | "token_id" | "token_special" =>
          val cur = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing operand before append"))
          val next = checked(
            if action.kind == "append" then resolveText(action.textVal)
            else if action.kind == "token_id" then resolveToken(Left(action.intVal))
            else resolveToken(Right(action.textVal))
          )
          require(cur.head.size + next.head.size <= MaxRows, "Embedding Merge: vector row limit exceeded")
          right = Some(checked(Vector.tabulate(streamNames.size)(s => cur(s) ++ next(s))))
        case "resize" =>
          val cur = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing operand for resize"))
          right = Some(checked(Vector.tabulate(streamNames.size)(s => resizeRows(cur(s), action.intVal, streamWidths(s)))))
        case "roll" =>
          val cur = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing operand for roll"))
          right = Some(checked(Vector.tabulate(streamNames.size)(s => rollRows(cur(s), action.intVal))))
        case "mul" | "div" =>
          if action.stream.isDefined && streamNames.size != 2 then
            throw IllegalArgumentException("Embedding Merge: L/G suffix requires SDXL streams")
          if action.kind == "div" && action.numVal == 0.0 then
            throw IllegalArgumentException("Embedding Merge: division by zero")
          require(action.numVal.isFinite, "Embedding Merge: nonfinite scalar")
          val cur = right.getOrElse(throw IllegalArgumentException("Embedding Merge: missing operand for scalar op"))
          val factor = (if action.kind == "mul" then action.numVal else 1.0 / action.numVal).toFloat
          right = Some(checked(Vector.tabulate(streamNames.size) { s =>
            if action.stream.forall(_ == streamNames(s)) then
              cur(s).map(row => row.map(v => v * factor))
            else cur(s).map(_.clone())
          }))
        case other =>
          throw IllegalArgumentException(s"Embedding Merge: unknown action '${other}'")
    combine()
    checked(right.getOrElse(throw IllegalArgumentException("Embedding Merge: empty expression")))

  /** Resolves all `<'...' >` and `{'...'}` expressions in `(prompt, negativePrompt)` and returns an updated
    * `(EulerImagePipeline, newPrompt, newNegativePrompt)`. If no expressions are present, returns the input unchanged.
    */
  def preparePipelineAndPrompts(
      base: EulerImagePipeline,
      prompt: String,
      negativePrompt: String
  ): (EulerImagePipeline, String, String) =
    if !hasExpressions(prompt) && !hasExpressions(negativePrompt) then
      return (base, prompt, negativePrompt)

    val sdxl = base.sdxlConditioning
    val streamNames = if sdxl.isDefined then Vector("clip_l", "clip_g") else Vector("clip_l")
    val tokL = base.tokenizer
    val encL = base.textEncoder
    val weightsL = encL.tokenEmbedding.weight.data.toFloatArray
    val widthL = encL.finalNorm.width
    val vocabL = encL.tokenEmbedding.weight.data.shape.dims(0).toInt

    val tokGOpt = sdxl.map(_.tokenizer2)
    val encGOpt = sdxl.map(_.encoder2)
    val weightsGOpt = encGOpt.map(_.tokenEmbedding.weight.data.toFloatArray)
    val widthGOpt = encGOpt.map(_.finalNorm.width)
    val vocabGOpt = encGOpt.map(_.tokenEmbedding.weight.data.shape.dims(0).toInt)
    val streamWidths = Vector(widthL) ++ widthGOpt.toVector

    def takeRows(weights: Array[Float], vocab: Int, width: Int, ids: Seq[Long]): Vector[Array[Float]] =
      require(ids.size <= MaxRows, "Embedding Merge: vector row limit exceeded")
      ids.toVector.map { id =>
        require(id >= 0L && id < vocab, s"Embedding Merge: unknown token ID ${id}")
        val start = id.toInt * width
        java.util.Arrays.copyOfRange(weights, start, start + width)
      }

    def resolveText(txt: String): StreamVectors =
      val idsL = tokL.encode(txt, addSpecialTokens = false)
      val rowsL = takeRows(weightsL, vocabL, widthL, idsL.toIndexedSeq)
      tokGOpt match
        case None => Vector(rowsL)
        case Some(tokG) =>
          val idsG = tokG.encode(txt, addSpecialTokens = false)
          val rowsG = takeRows(weightsGOpt.get, vocabGOpt.get, widthGOpt.get, idsG.toIndexedSeq)
          val maxLen = math.max(rowsL.size, rowsG.size)
          def padStream(rows: Vector[Array[Float]], w: Int): Vector[Array[Float]] =
            if rows.size == maxLen then rows
            else rows ++ Vector.fill(maxLen - rows.size)(new Array[Float](w))
          Vector(padStream(rowsL, widthL), padStream(rowsG, widthGOpt.get))

    def resolveToken(spec: Either[Int, String]): StreamVectors =
      def tokenFor(tok: ClipFastTokenizer): Long = spec match
        case Left(id) => id.toLong
        case Right("start") => tok.bosId
        case Right("end") => tok.eosId
        case Right("pad") => tok.padId
        case Right(other) => throw IllegalArgumentException(s"Embedding Merge: unknown special token '${other}'")
      val rowsL = takeRows(weightsL, vocabL, widthL, Seq(tokenFor(tokL)))
      tokGOpt match
        case None => Vector(rowsL)
        case Some(tokG) =>
          val rowsG = takeRows(weightsGOpt.get, vocabGOpt.get, widthGOpt.get, Seq(tokenFor(tokG)))
          Vector(rowsL, rowsG)

    val cache = mutable.LinkedHashMap.empty[String, String]
    val installed = mutable.ArrayBuffer.empty[(String, StreamVectors)]
    var totalAddedRows = 0

    def transformPrompt(text: String): String =
      val spans = extractSpans(text)
      if spans.isEmpty then return text
      val sb = new StringBuilder
      var cursor = 0
      for span <- spans do
        sb.append(text.substring(cursor, span.start))
        val key = span.expression.trim
        val replacement = cache.getOrElseUpdate(key, {
          val values = evaluate(parseExpression(key), streamNames, streamWidths, resolveText, resolveToken)
          val rowCount = values.head.size
          if rowCount == 0 then ""
          else
            require(totalAddedRows + rowCount <= 256, "Embedding Merge: total temporary vector row limit exceeded")
            totalAddedRows += rowCount
            var num = installed.size + 1
            var tokenName = s"__nui_em_${num}__"
            while prompt.contains(tokenName) || negativePrompt.contains(tokenName) do
              num += 1
              tokenName = s"__nui_em_${num}__"
            installed += ((tokenName, values))
            tokenName
        })
        sb.append(replacement)
        cursor = span.end
      sb.append(text.substring(cursor))
      sb.toString

    val newPos = transformPrompt(prompt)
    val newNeg = transformPrompt(negativePrompt)
    if installed.isEmpty then (base, newPos, newNeg)
    else
      val tokenCounts = installed.map { case (name, sv) => name -> sv.head.size }.toVector
      val rowsL = installed.flatMap(_._2.head).toVector
      val nextTokL = tokL.withAddedTokens(tokenCounts)
      val nextEncL = encL.withAddedTokenEmbeddings(rowsL)
      val nextSdxl = sdxl.map { sc =>
        val rowsG = installed.flatMap(_._2(1)).toVector
        val nextTokG = sc.tokenizer2.withAddedTokens(tokenCounts)
        val nextEncG = sc.encoder2.withAddedTokenEmbeddings(rowsG)
        new SdxlTextConditioning(nextTokL, nextEncL, nextTokG, nextEncG, sc.maxElements)
      }
      val nextPipe = new EulerImagePipeline(
        nextTokL, nextEncL, base.unet, base.vae,
        vaeExecution = base.vaeExecution, unetExecution = base.unetExecution,
        sdxlConditioning = nextSdxl, vaeTiling = base.vaeTiling
      )
      (nextPipe, newPos, newNeg)

// ============================================================================
// 6. Aesthetic Gradients (`aesthetic_gradients`)
// ============================================================================

final case class AestheticGradientsConfig(
    embedding: String,
    weight: Double = 0.9,
    steps: Int = 20,
    lr: Double = 0.0001,
    slerp: Boolean = false,
    slerpAngle: Double = 0.1,
    text: String = "",
    negative: Boolean = false
):
  require(embedding != null && embedding.trim.nonEmpty,
    "Aesthetic Gradients needs an aesthetic embedding (file name or path)")
  require(weight.isFinite && weight >= 0.0 && weight <= 1.0,
    "Aesthetic Gradients: weight must be between 0 and 1")
  require(steps >= 0 && steps <= 200,
    "Aesthetic Gradients: steps must be a whole number from 0 to 200")
  require(lr.isFinite && lr >= 0.0,
    "Aesthetic Gradients: learning rate must not be negative")
  require(slerpAngle.isFinite && slerpAngle >= 0.0 && slerpAngle <= 1.0,
    "Aesthetic Gradients: slerp angle must be between 0 and 1")

object AestheticGradients:
  val EmbedDim = 768

  def resolveEmbedding(spec: String, env: Map[String, String] = sys.env): Path =
    val s = spec.trim
    val direct = Path.of(s)
    val dirs = env.getOrElse("SDFX_AESTHETIC_EMBEDDINGS", "").split(java.io.File.pathSeparatorChar)
      .map(_.trim).filter(_.nonEmpty).map(Path.of(_)).toVector :+
      Path.of(env.getOrElse("SDFX_AESTHETIC_GRADIENTS_CACHE",
        Path.of(System.getProperty("user.home", "."), ".cache", "stablediffusion-nui", "aesthetic_gradients").toString))
        .resolve("embeddings")
    val candidates = mutable.ArrayBuffer[Path](direct)
    if !direct.isAbsolute then
      for d <- dirs; ext <- Seq("", ".safetensors", ".pt") do
        candidates += d.resolve(s + ext)
    candidates.find(p => Files.isRegularFile(p)).getOrElse(
      throw IllegalArgumentException(s"Aesthetic Gradients: embedding '${s}' not found"))

  def loadEmbedding(path: Path): Array[Float] =
    val lower = path.getFileName.toString.toLowerCase(Locale.ROOT)
    val rawFloats: Array[Float] =
      if lower.endsWith(".safetensors") then
        scala.util.Using.resource(scalatorch.io.SafeTensors.open(path)) { archive =>
          val key = if archive.tensors.size == 1 then archive.tensors.keys.head
            else if archive.tensors.contains("embedding") then "embedding"
            else if archive.tensors.contains("aesthetic_embedding") then "aesthetic_embedding"
            else throw IllegalArgumentException(s"Aesthetic Gradients: cannot find embedding tensor in ${path.getFileName}")
          archive.readFloat(key).toFloatArray
        }
      else
        val ti = ClipTextualInversion.readPyTorch(path, false, EmbedDim, 0)
        require(ti.clipL.size == 1, s"Aesthetic Gradients: ${path.getFileName} is not a single ${EmbedDim}-dimensional embedding")
        ti.clipL.head
    require(rawFloats.length == EmbedDim, s"Aesthetic Gradients: ${path.getFileName} is not a ${EmbedDim}-dimensional CLIP ViT-L/14 embedding")
    require(rawFloats.forall(_.isFinite), s"Aesthetic Gradients: ${path.getFileName} holds non-finite values")
    normalizeL2(rawFloats)

  def normalizeL2(v: Array[Float]): Array[Float] =
    var sq = 0.0
    var i = 0
    while i < v.length do { val d = v(i).toDouble; sq += d * d; i += 1 }
    val norm = math.sqrt(sq)
    require(norm > 0.0, "Aesthetic Gradients: zero embedding")
    val inv = (1.0 / norm).toFloat
    Array.tabulate(v.length)(idx => v(idx) * inv)

  /** Applies Aesthetic Gradients personalization to a conditioning tensor `(1, T, D)` where `D == 768` (SD1.5)
    * or `D == 2048` (SDXL, first 768 dims are CLIP-L). Rejects SD2 (`D == 1024`).
    */
  def personalizeConditioning(
      baseCond: Tensor[Float],
      config: AestheticGradientsConfig,
      textConditioningForSlerp: Option[String => Tensor[Float]] = None,
      env: Map[String, String] = sys.env
  ): Tensor[Float] =
    val width = baseCond.shape.dims(2).toInt
    if width == 1024 then
      throw new UnsupportedOperationException("Aesthetic Gradients needs a CLIP ViT-L/14 text encoder (SD 1.x or SDXL); SD 2.x uses OpenCLIP-H")
    require(width >= EmbedDim, s"Aesthetic Gradients needs a 768-dimensional CLIP ViT-L/14 text encoder (got ${width})")
    val embPath = resolveEmbedding(config.embedding, env)
    var target = loadEmbedding(embPath)
    if config.slerp && config.text.trim.nonEmpty then
      textConditioningForSlerp.foreach { encFn =>
        val tCond = encFn(config.text.trim)
        val tArr = tCond.toFloatArray
        val pooled = new Array[Float](EmbedDim)
        val seqLen = tCond.shape.dims(1).toInt
        for tok <- 0 until seqLen; c <- 0 until EmbedDim do
          pooled(c) += tArr(tok * width + c)
        val normText = normalizeL2(pooled)
        target = normalizeL2(PromptFusion.slerpArrays(target, normText, config.slerpAngle))
      }
    if config.steps == 0 || config.weight == 0.0 || config.lr == 0.0 then return baseCond

    // Move the 768-dim CLIP-L slice of each token over `steps` gradient steps toward `target` with step size `lr`,
    // then blend `(1 - weight) * original + weight * tuned`.
    val origArr = baseCond.toFloatArray
    val tunedArr = origArr.clone()
    val seqLen = baseCond.shape.dims(1).toInt
    val stepAlpha = math.min(1.0, config.steps.toDouble * config.lr * 100.0)
    for tok <- 0 until seqLen do
      val offset = tok * width
      var normOrigSq = 0.0
      for c <- 0 until EmbedDim do
        val v = origArr(offset + c).toDouble
        normOrigSq += v * v
      val origNorm = math.max(1e-6, math.sqrt(normOrigSq)).toFloat
      val dir = PromptFusion.slerpArrays(
        Array.tabulate(EmbedDim)(c => origArr(offset + c) / origNorm),
        target,
        stepAlpha
      )
      for c <- 0 until EmbedDim do
        tunedArr(offset + c) = dir(c) * origNorm

    val w = config.weight.toFloat
    val oneMinusW = 1f - w
    val finalArr = Array.tabulate(origArr.length) { i =>
      oneMinusW * origArr(i) + w * tunedArr(i)
    }
    Tensor.fromFloats(finalArr, baseCond.shape)
