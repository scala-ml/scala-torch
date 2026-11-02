package scalatorch.nn

import scalatorch.*
import scalatorch.autograd.*
import java.nio.file.{Files, Path}

class CheckpointSuite extends munit.FunSuite:
  case class Model(layer: Linear) derives Module
  case class Tied(a: Parameter, b: Parameter) derives Module
  private def scalar(x: Float) = Tensor.fromFloats(Array(x), Shape.from(Seq.empty))
  private def same[A: Module](a: A, b: A): Unit =
    val left = Module.stateDict(a)
    val right = Module.stateDict(b)
    assertEquals(left.keySet, right.keySet)
    left.foreach((name, t) => assertEquals(t.toFloatArray.toSeq, right(name).toFloatArray.toSeq))
  private def file[A](f: Path => A): A =
    val path = Files.createTempFile("scala-torch-test-", ".stc")
    try f(path)
    finally Files.deleteIfExists(path)
  private def train(model: Model, optimizer: Optimizer, steps: Int): Unit =
    val x = Variable(Tensor.fromFloats(Array(-1f, 0f, 1f, 2f), Shape.from(Seq(4L, 1L))), false)
    val target = Variable(Tensor.fromFloats(Array(-1f, 1f, 3f, 5f), Shape.from(Seq(4L, 1L))), false)
    for _ <- 0 until steps do
      optimizer.zeroGrad()
      val residual = model.layer(x) - target
      (residual * residual).mean().backward()
      optimizer.step()

  test("AdamW checkpoint resumes weights moments and counters exactly") { file { path =>
    val full = Model(Linear.init(1, 1, seed = 3))
    val split = Model(Linear.init(1, 1, seed = 3))
    val a = AdamW(Module.parameters(full), 0.03)
    val b = AdamW(Module.parameters(split), 0.03)
    train(full, a, 40)
    train(split, b, 17)
    Checkpoint.save(split, b, path, step = 17)
    val resumed = Model(Linear.init(1, 1, seed = 99))
    val c = AdamW(Module.parameters(resumed), 0.03)
    assertEquals(Checkpoint.load(resumed, c, path), 17L)
    assert(Module.parameters(resumed).forall(_.grad.isEmpty))
    train(resumed, c, 23)
    same(full, resumed)
  }}
  test("SGD tied parameters and optimizer subset restore by parameter identity") { file { path =>
    val p = Parameter(scalar(2))
    val model = Tied(p, p)
    val optimizer = SGD(Seq(p), 0.1)
    (p.value * p.value).backward(); optimizer.step()
    Checkpoint.save(model, optimizer, path, step = 1)
    val q = Parameter(scalar(9))
    val copy = Tied(q, q)
    val restored = SGD(Seq(q), 0.1)
    assertEquals(Checkpoint.load(copy, restored, path), 1L)
    same(model, copy)
    assert(q.grad.isEmpty)
  }}
  test("corrupt truncated and trailing checkpoint payloads leave destination unchanged") { file { path =>
    val source = Model(Linear.init(1, 1, seed = 1))
    val optimizer = AdamW(Module.parameters(source), 0.03)
    train(source, optimizer, 3)
    Checkpoint.save(source, optimizer, path, 3)
    val bytes = Files.readAllBytes(path)
    val target = Model(Linear.init(1, 1, seed = 5))
    val control = Model(Linear.init(1, 1, seed = 5))
    val targetOptimizer = AdamW(Module.parameters(target), 0.03)
    val bad = bytes.clone(); bad(bad.length - 1) = (bad.last ^ 1).toByte
    val oversizedText = bytes.clone()
    java.nio.ByteBuffer.wrap(oversizedText).putInt(16, Int.MaxValue)
    for contents <- Seq(bad, bytes.take(bytes.length / 2), bytes ++ Array[Byte](0), Array.fill[Byte](80)(-1), oversizedText) do
      Files.write(path, contents)
      intercept[IllegalArgumentException](Checkpoint.load(target, targetOptimizer, path))
      same(target, control)
    // A rejected restore must also leave the optimizer pristine.
    train(target, targetOptimizer, 2)
    train(control, AdamW(Module.parameters(control), 0.03), 2)
    same(target, control)
  }}
  test("configuration and alias mismatches reject before weight updates") { file { path =>
    val source = Model(Linear.init(1, 1, seed = 2))
    Checkpoint.save(source, SGD(Module.parameters(source), 0.1), path, 0)
    val target = Model(Linear.init(1, 1, seed = 5))
    val control = Model(Linear.init(1, 1, seed = 5))
    intercept[IllegalArgumentException](Checkpoint.load(target, SGD(Module.parameters(target), 0.2), path))
    intercept[IllegalArgumentException](Checkpoint.load(target, AdamW(Module.parameters(target), 0.1), path))
    same(target, control)
    val p = Parameter(scalar(2))
    val tied = Tied(p, p)
    Checkpoint.save(tied, SGD(Seq(p), 0.1), path, 0)
    val untied = Tied(Parameter(scalar(3)), Parameter(scalar(4)))
    intercept[IllegalArgumentException](Checkpoint.load(untied, SGD(Module.parameters(untied), 0.1), path))
    assertEquals(untied.a.data.toFloatArray.toSeq, Seq(3f))
  }}
  test("failed save keeps existing checkpoint and unrelated optimizer is rejected") { file { path =>
    val model = Model(Linear.init(1, 1, seed = 1))
    val optimizer = SGD(Module.parameters(model), 0.1)
    Checkpoint.save(model, optimizer, path, 0)
    val original = Files.readAllBytes(path).toSeq
    intercept[IllegalArgumentException](Checkpoint.save(model, optimizer, path, -1))
    intercept[IllegalArgumentException](Checkpoint.save(model, SGD(Seq(Parameter(scalar(1))), 0.1), path, 0))
    assertEquals(Files.readAllBytes(path).toSeq, original)
  }}
  test("missing Adam slots and reordered optimizer parameters resume by names") { file { path =>
    val source = Tied(Parameter(scalar(1)), Parameter(scalar(2)))
    val a = AdamW(Seq(source.b, source.a), 0.1)
    (source.a.value * 2f).backward(); a.step()
    Checkpoint.save(source, a, path, 1)
    val target = Tied(Parameter(scalar(8)), Parameter(scalar(9)))
    val b = AdamW(Seq(target.a, target.b), 0.1)
    Checkpoint.load(target, b, path)
    for (model, optimizer) <- Seq(source -> a, target -> b) do
      optimizer.zeroGrad()
      (model.a.value * 4f + model.b.value * 3f).backward()
      optimizer.step()
    same(source, target)
  }}
  test("optimizer subsets must match and Float32 payload bits survive") { file { path =>
    val bits = Array(0x7fc01234, 0x80000000, 0x7f800000)
    def payload = Tensor.fromFloats(bits.map(java.lang.Float.intBitsToFloat), Shape.from(Seq(3L)))
    val source = Tied(Parameter(payload), Parameter(scalar(2)))
    Checkpoint.save(source, SGD(Seq(source.b), 0.1), path, 0)
    val target = Tied(Parameter(Tensor.fromFloats(Array(0f, 0f, 0f), Shape.from(Seq(3L)))), Parameter(scalar(9)))
    intercept[IllegalArgumentException](Checkpoint.load(target, SGD(Module.parameters(target), 0.1), path))
    assertEquals(target.b.data.toFloatArray.toSeq, Seq(9f))
    Checkpoint.load(target, SGD(Seq(target.b), 0.1), path)
    assertEquals(target.a.data.toFloatArray.map(java.lang.Float.floatToRawIntBits).toSeq, bits.toSeq)
  }}
  test("failed atomic replacement cleans the temporary file and preserves directory contents") {
    val root = Files.createTempDirectory("scala-torch-save-failure-")
    val destination = Files.createDirectory(root.resolve("checkpoint"))
    val sentinel = Files.write(destination.resolve("keep"), Array[Byte](1, 2, 3))
    try
      val model = Model(Linear.init(1, 1, seed = 1))
      intercept[java.io.IOException](Checkpoint.save(model, SGD(Module.parameters(model), 0.1), destination, 0))
      assertEquals(Files.readAllBytes(sentinel).toSeq, Seq[Byte](1, 2, 3))
      val entries = Files.list(root)
      try assertEquals(entries.count(), 1L)
      finally entries.close()
    finally
      Files.deleteIfExists(sentinel)
      Files.deleteIfExists(destination)
      Files.deleteIfExists(root)
  }
  test("valid-checksum files with invalid Adam counters or moments reject before commit") { file { path =>
    val p = Parameter(scalar(1))
    val source = Tied(p, p)
    val optimizer = AdamW(Seq(p), 0.1)
    (p.value * 2f).backward(); optimizer.step()
    Checkpoint.save(source, optimizer, path, 1)
    val original = Files.readAllBytes(path)
    val raw = new java.io.ByteArrayInputStream(original)
    val in = new java.io.DataInputStream(raw)
    def skipText(): Unit = in.skipNBytes(in.readInt().toLong)
    in.skipNBytes(16) // magic + caller step
    skipText() // optimizer kind
    val configCount = in.readInt()
    for _ <- 0 until configCount do { skipText(); in.readDouble() }
    assertEquals(in.readInt(), 1) // one identity with two aliases
    val aliasCount = in.readInt()
    for _ <- 0 until aliasCount do skipText()
    in.readBoolean()
    assertEquals(in.readInt(), 0) // scalar, so no dimension words
    in.readInt() // weight
    assert(in.readBoolean())
    assert(in.readBoolean())
    val counterOffset = original.length - raw.available()
    val q = Parameter(scalar(4))
    val target = Tied(q, q)
    val restored = AdamW(Seq(q), 0.1)
    (q.value * 3f).backward()
    val mutations: Seq[java.nio.ByteBuffer => Unit] = Seq(
      b => { b.putLong(counterOffset, 0L); () },
      b => { b.putDouble(counterOffset + 8, Double.NaN); () },
      b => { b.putDouble(counterOffset + 16, -1.0); () })
    mutations.foreach { mutate =>
      val bytes = original.clone()
      mutate(java.nio.ByteBuffer.wrap(bytes))
      val digest = java.security.MessageDigest.getInstance("SHA-256").digest(bytes.dropRight(32))
      System.arraycopy(digest, 0, bytes, bytes.length - 32, 32)
      Files.write(path, bytes)
      intercept[IllegalArgumentException](Checkpoint.load(target, restored, path))
      assertEquals(q.data.toFloatArray.toSeq, Seq(4f))
      assertEquals(q.grad.get.toFloatArray.toSeq, Seq(3f))
    }
    restored.step()
    assertEqualsDouble(q.data.toFloatArray(0), 3.896, 1e-6) // pristine first AdamW step
  }}
