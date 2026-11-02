package scalatorch.models

import munit.FunSuite
import java.nio.file.{Files, Path}
import java.nio.charset.StandardCharsets.UTF_8
import java.security.MessageDigest
import com.fasterxml.jackson.core.io.JsonStringEncoder

class TokenizerSuite extends FunSuite:
  private def quote(s: String): String = "\"" + new String(JsonStringEncoder.getInstance().quoteAsString(s)) + "\""
  private val alphabet =
    var extra = 256
    (0 to 255).map { b =>
      if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 then b.toChar.toString
      else { val c = extra.toChar.toString; extra += 1; c }
    }
  private def json(merges: String = "[\"a b\",\"ab a\"]", digits: Boolean = false): String =
    val vocab = (alphabet ++ Vector("ab", "aba", "<s>")).zipWithIndex.map((s,i) => s"${quote(s)}:$i").mkString(",")
    val byte = """{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}"""
    val pre = if digits then s"""{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},$byte]}""" else byte
    s"""{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"post_processor":null,"pre_tokenizer":$pre,"decoder":$byte,"added_tokens":[{"id":258,"content":"<s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{$vocab},"merges":$merges}}"""

  test("ranked merges and overlapping pairs produce exact IDs") {
    val t = ByteBpeTokenizer.fromJson(json())
    assertEquals(t.vocabSize, 259)
    assertEquals(t.encode("ababa").toVector, Vector(256L,257L))
    assertEquals(t.decode(t.encode("ababa")), "ababa")
    assertEquals(t.encode("").toVector, Vector.empty)
  }
  test("byte alphabet roundtrips Unicode whitespace controls and embedded special tokens") {
    val t = ByteBpeTokenizer.fromJson(json(digits=true))
    val text = " 日本語\u0000\t\r\n123４²½ 🦀 e\u0301<s><s>x"
    val ids = t.encode(text)
    assertEquals(t.decode(ids, false), text)
    assertEquals(t.decode(ids), text.replace("<s>", ""))
    assertEquals(ids.count(_ == 258L), 2)
    assertEquals(t.decode(Array(0xc3L)), "�")
    assertEquals(t.decode(Array(0xedL, 0xa0L, 0x80L)), "���")
    intercept[IllegalArgumentException](t.decode(Array(-1L)))
    intercept[IllegalArgumentException](t.decode(Array(259L)))
    intercept[IllegalArgumentException](t.encode("\ud800"))
    intercept[IllegalArgumentException](t.encode("x" * 1048577))
    intercept[IllegalArgumentException](t.encode("x" * 16385))
  }
  test("absent byte symbols follow no-UNK BPE omission semantics") {
    val t = ByteBpeTokenizer.fromJson(json().replace("\"Ą\":4", "\"unused\":4"))
    assertEquals(t.decode(t.encode("a\u0004b")), "ab")
  }
  test("unsupported pipelines and malformed vocabulary or merge metadata are rejected") {
    val base = json()
    val bad = Vector(
      base.replace("\"normalizer\":null", "\"normalizer\":{\"type\":\"Lowercase\"}"),
      base.replace("\"post_processor\":null", "\"post_processor\":{\"type\":\"TemplateProcessing\"}"),
      base.replace("\"byte_fallback\":false", "\"byte_fallback\":true"),
      base.replace("\"dropout\":null", "\"dropout\":0.1"),
      base.replace("\"lstrip\":false", "\"lstrip\":true"),
      base.replace("\"add_prefix_space\":false", "\"add_prefix_space\":true"),
      base.replace("\"use_regex\":true", "\"use_regex\":false"),
      base.replace("\"ab\":256", "\"ab\":255"),
      base.replace("\"ab\":256", "\"ab\":256.0"),
      base.replace("\"ab\":256", "\"ab\":256,\"ab\":256"),
      json("[\"a b\",\"a b\"]"), json("[\"a z\"]"),
      base + " {}", "{", base.replace("\"version\":\"1.0\"", "\"version\":\"2.0\"")
    )
    bad.foreach(s => intercept[IllegalArgumentException](ByteBpeTokenizer.fromJson(s)))
  }
  test("pinned SmolLM2 matches Rust tokenizers IDs and both decode modes") {
    val dir = sys.props.get("scalatorch.smollm.dir").getOrElse(assume(false, "set scalatorch.smollm.dir for public tokenizer gate").toString)
    val path = Path.of(dir, "tokenizer.json")
    val hash = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).map(b => f"${b & 255}%02x").mkString
    assertEquals(hash, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c")
    val t = ByteBpeTokenizer.read(path)
    def unhex(s: String): String = new String(s.grouped(2).map(Integer.parseInt(_,16).toByte).toArray, UTF_8)
    val source = scala.io.Source.fromResource("tokenizer/smollm2.tsv")
    val lines = source.getLines().toVector
    assertEquals(lines.size, 13)
    try lines.foreach { line =>
      val fields = line.split("\t", -1)
      val ids = if fields(1).isEmpty then Array.emptyLongArray else fields(1).split(",").map(_.toLong)
      assertEquals(t.encode(unhex(fields(0))).toVector, ids.toVector, clues(unhex(fields(0))))
      assertEquals(t.decode(ids, false), unhex(fields(2)))
      assertEquals(t.decode(ids, true), unhex(fields(3)))
    } finally source.close()
    val decoded = scala.io.Source.fromResource("tokenizer/smollm2-decode.tsv")
    val decodeLines = decoded.getLines().toVector
    assertEquals(decodeLines.size, 7)
    try decodeLines.foreach { line =>
      val fields = line.split("\\t", -1)
      assertEquals(t.decode(fields(0).split(",").map(_.toLong), false), unhex(fields(1)))
    } finally decoded.close()
  }
