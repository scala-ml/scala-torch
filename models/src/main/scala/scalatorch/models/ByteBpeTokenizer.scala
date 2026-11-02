package scalatorch.models

import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer
import TokenizerJson.Value

/** Supported HF byte-level BPE subset. Immutable; no automatic BOS/EOS or chat formatting. */
final class ByteBpeTokenizer private (
    vocabulary: Map[String, Int], tokens: Vector[String], ranks: Map[(String, String), Int],
    special: Map[String, Int], individualDigits: Boolean):
  import ByteBpeTokenizer.*
  val vocabSize: Int = tokens.size
  private val specialIds = special.values.toSet
  private val specialPattern = Option.when(special.nonEmpty)(Pattern.compile(
    special.keys.toVector.sortBy(s => (-s.length, s)).map(Pattern.quote).mkString("|")))

  def encode(text: String): Array[Long] =
    require(text.length <= 1048576 && TokenizerJson.validUnicode(text), "invalid or oversized tokenizer input")
    val output = ArrayBuffer.empty[Long]
    def piece(text: String): Unit =
      val bytes = text.getBytes(UTF_8)
      require(bytes.length <= 16384, "BPE piece exceeds 16384 bytes")
      val symbols = ArrayBuffer.from(bytes.iterator.map(b => alphabet(b & 255)).filter(vocabulary.contains))
      var merging = true
      while merging && symbols.length > 1 do
        var bestRank = Int.MaxValue
        var bestIndex = -1
        var i = 0
        while i + 1 < symbols.length do
          ranks.get((symbols(i), symbols(i + 1))).foreach { rank =>
            if rank < bestRank then { bestRank = rank; bestIndex = i }
          }
          i += 1
        if bestIndex < 0 then merging = false
        else
          symbols(bestIndex) = symbols(bestIndex) + symbols(bestIndex + 1)
          symbols.remove(bestIndex + 1)
      symbols.foreach(s => output += vocabulary(s).toLong)
    def byteLevel(text: String): Unit =
      val matcher = bytePattern.matcher(text)
      while matcher.find() do piece(matcher.group())
    def ordinary(text: String): Unit =
      if individualDigits then
        val matcher = numberPattern.matcher(text)
        var start = 0
        while matcher.find() do
          byteLevel(text.substring(start, matcher.start()))
          byteLevel(matcher.group())
          start = matcher.end()
        byteLevel(text.substring(start))
      else byteLevel(text)
    specialPattern match
      case None => ordinary(text)
      case Some(pattern) =>
        val matcher = pattern.matcher(text)
        var start = 0
        while matcher.find() do
          ordinary(text.substring(start, matcher.start()))
          output += special(matcher.group()).toLong
          start = matcher.end()
        ordinary(text.substring(start))
    output.toArray

  def decode(ids: Array[Long], skipSpecialTokens: Boolean = true): String =
    require(ids.length <= 1048576, "too many tokenizer IDs")
    val output = new ByteArrayOutputStream()
    ids.foreach { id =>
      require(id >= 0 && id < tokens.length, s"token ID out of vocabulary: $id")
      if !(skipSpecialTokens && specialIds(id.toInt)) then
        val token = tokens(id.toInt)
        val bytes = if token.forall(inverseAlphabet.contains) then token.iterator.map(c => inverseAlphabet(c).toByte).toArray
          else token.getBytes(UTF_8)
        require(output.size().toLong + bytes.length <= TokenizerJson.maxBytes, "decoded text exceeds 32MiB")
        output.write(bytes)
    }
    decodeUtf8(output.toByteArray)

object ByteBpeTokenizer:
  private val alphabet: Vector[String] =
    var next = 256
    (0 to 255).map { b =>
      if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 then b.toChar.toString
      else { val s = next.toChar.toString; next += 1; s }
    }.toVector
  private val inverseAlphabet = alphabet.zipWithIndex.map((s,i) => s.charAt(0) -> i).toMap
  private val bytePattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", Pattern.UNICODE_CHARACTER_CLASS)
  private val numberPattern = Pattern.compile("\\p{N}")

  // Rust String::from_utf8_lossy uses maximal valid prefixes of malformed sequences.
  // Java's decoder combines surrogate encodings into one replacement instead.
  private def decodeUtf8(bytes: Array[Byte]): String =
    val out = new java.lang.StringBuilder()
    var i = 0
    while i < bytes.length do
      val first = bytes(i) & 255
      val width = if first < 128 then 1 else if first >= 194 && first <= 223 then 2
        else if first >= 224 && first <= 239 then 3 else if first >= 240 && first <= 244 then 4 else 0
      if width == 1 then { out.append(first.toChar); i += 1 }
      else if width == 0 then { out.append('�'); i += 1 }
      else
        var consumed = 1
        var cp = first & (if width == 2 then 31 else if width == 3 then 15 else 7)
        var valid = true
        while consumed < width && valid do
          if i + consumed == bytes.length then valid = false
          else
            val b = bytes(i + consumed) & 255
            val min = if consumed == 1 && first == 224 then 160 else if consumed == 1 && first == 240 then 144 else 128
            val max = if consumed == 1 && first == 237 then 159 else if consumed == 1 && first == 244 then 143 else 191
            if b < min || b > max then valid = false
            else { cp = (cp << 6) | (b & 63); consumed += 1 }
        if valid then out.appendCodePoint(cp) else out.append('�')
        i += consumed
    out.toString

  def read(path: Path): ByteBpeTokenizer = build(TokenizerJson.read(path))
  def fromJson(json: String): ByteBpeTokenizer = build(TokenizerJson.parse(json))

  private def build(value: Value): ByteBpeTokenizer =
    import Value.*
    def get(fields: Map[String, Value], key: String): Value =
      fields.getOrElse(key, throw IllegalArgumentException(s"missing tokenizer field: $key"))
    def exact(fields: Map[String, Value], key: String, expected: Value): Unit =
      require(get(fields, key) == expected, s"unsupported tokenizer option: $key")
    def allowed(fields: Map[String, Value], keys: String*): Unit =
      require(fields.keySet.subsetOf(keys.toSet), s"unknown tokenizer fields: ${fields.keySet -- keys.toSet}")
    val root = value.obj
    allowed(root, "version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder", "model")
    exact(root, "version", Str("1.0"))
    for key <- Seq("truncation", "padding", "normalizer", "post_processor") do exact(root, key, Null)
    val decoder = get(root, "decoder").obj
    allowed(decoder, "type", "add_prefix_space", "trim_offsets", "use_regex")
    exact(decoder, "type", Str("ByteLevel"))
    def byteLevel(v: Value): Unit =
      val fields = v.obj
      allowed(fields, "type", "add_prefix_space", "trim_offsets", "use_regex")
      exact(fields, "type", Str("ByteLevel"))
      exact(fields, "add_prefix_space", Bool(false))
      exact(fields, "use_regex", Bool(true))
      require(get(fields, "trim_offsets").isInstanceOf[Bool], "invalid trim_offsets")
    val pre = get(root, "pre_tokenizer")
    val digits = pre.obj.get("type").contains(Str("Sequence"))
    if digits then
      allowed(pre.obj, "type", "pretokenizers")
      val stages = get(pre.obj, "pretokenizers").arr
      require(stages.size == 2, "unsupported tokenizer sequence")
      require(stages.head == Obj(Map("type" -> Str("Digits"), "individual_digits" -> Bool(true))), "unsupported Digits stage")
      byteLevel(stages(1))
    else byteLevel(pre)
    val model = get(root, "model").obj
    allowed(model, "type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk", "byte_fallback", "ignore_merges", "vocab", "merges")
    exact(model, "type", Str("BPE"))
    for key <- Seq("dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix") do exact(model, key, Null)
    for key <- Seq("fuse_unk", "byte_fallback", "ignore_merges") do
      require(model.getOrElse(key, Bool(false)) == Bool(false), s"unsupported BPE option: $key")
    val vocab = get(model, "vocab").obj.map((s,id) => s -> id.int)
    require(vocab.size >= 256 && vocab.size <= 1000000 && !vocab.contains(""), "invalid BPE vocabulary size/content")
    require(vocab.values.toSet == (0 until vocab.size).toSet, "vocabulary IDs must be unique and contiguous")
    val tokens = vocab.toVector.sortBy(_._2).map(_._1)
    val added = get(root, "added_tokens").arr.map { entry =>
      val fields = entry.obj
      allowed(fields, "id", "content", "single_word", "lstrip", "rstrip", "normalized", "special")
      for key <- Seq("single_word", "lstrip", "rstrip", "normalized") do exact(fields, key, Bool(false))
      exact(fields, "special", Bool(true))
      val content = get(fields, "content").str
      val id = get(fields, "id").int
      require(content.nonEmpty && vocab.get(content).contains(id), "added token must match vocabulary")
      content -> id
    }
    require(added.map(_._1).distinct.size == added.size, "duplicate added token")
    val special = added.toMap
    require(vocab.keys.forall(s => special.contains(s) || s.forall(inverseAlphabet.contains)), "vocabulary token outside byte alphabet")
    val mergeEntries = get(model, "merges").arr
    require(mergeEntries.size <= 1000000, "too many BPE merges")
    val merges = mergeEntries.zipWithIndex.map { (v,rank) =>
      val pair = v match
        case Str(s) => s.split(" ", -1).toVector
        case Arr(values) => values.map(_.str)
        case _ => throw IllegalArgumentException("invalid BPE merge")
      require(pair.size == 2 && pair.forall(s => vocab.contains(s) && !special.contains(s)) &&
        vocab.contains(pair.mkString) && !special.contains(pair.mkString), "BPE merge missing vocabulary token")
      (pair(0), pair(1)) -> rank
    }
    require(merges.map(_._1).distinct.size == merges.size, "duplicate BPE merge")
    new ByteBpeTokenizer(vocab, tokens, merges.toMap, special, digits)
