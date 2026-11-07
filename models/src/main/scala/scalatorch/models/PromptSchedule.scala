package scalatorch.models

import java.util.regex.Pattern

object PromptSchedule:
  private val NumberPattern = Pattern.compile("""[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?""")
  private val WsPattern = Pattern.compile("""\s+""")
  private val PlainPattern = Pattern.compile("""(?:[^\\\[\]():|]|\\.)+""")

  private sealed trait Node
  private final case class Lit(text: String) extends Node
  private final case class Group(items: Vector[Node]) extends Node
  private final class Sched(val before: Option[Vector[Node]], val after: Vector[Node], val when: String, var step: Int = 0) extends Node
  private final case class Alt(options: Vector[Vector[Node]]) extends Node

  private final class Fail extends RuntimeException(null, null, false, false)

  private final class Parser(s: String):
    private def matchAt(p: Pattern, idx: Int): Option[(String, Int)] =
      val m = p.matcher(s)
      m.region(idx, s.length)
      if m.lookingAt() then Some((m.group(0), m.end())) else None

    private def expect(i: Int, ch: Char): Int =
      if i < s.length && s.charAt(i) == ch then i + 1
      else throw new Fail

    private def numberClose(start: Int): (String, Int) =
      var i = matchAt(WsPattern, start).fold(start)(_._2)
      val (num, afterNum) = matchAt(NumberPattern, i).getOrElse(throw new Fail)
      i = matchAt(WsPattern, afterNum).fold(afterNum)(_._2)
      (num, expect(i, ']'))

    def prompt(start: Int): (Vector[Node], Int) =
      val out = Vector.newBuilder[Node]
      var i = start
      var loop = true
      while i < s.length && loop do
        val c = s.charAt(i)
        if c == '(' || c == '[' then
          try
            val (node, next) = bracket(i)
            out += node
            i = next
          catch case _: Fail => loop = false
        else
          matchAt(PlainPattern, i) match
            case Some((txt, next)) =>
              out += Lit(txt)
              i = next
            case None =>
              loop = false
      (out.result(), i)

    def bracket(i: Int): (Node, Int) =
      if s.charAt(i) == '(' then
        val (p1, j) = prompt(i + 1)
        if j < s.length && s.charAt(j) == ')' then
          (Group(Vector(Lit("("), Group(p1), Lit(")"))), j + 1)
        else if j < s.length && s.charAt(j) == ':' then
          val (p2, k) = prompt(j + 1)
          (Group(Vector(Lit("("), Group(p1), Lit(":"), Group(p2), Lit(")"))), expect(k, ')'))
        else throw new Fail
      else
        val (p1, j) = prompt(i + 1)
        if j >= s.length then throw new Fail
        val c = s.charAt(j)
        if c == ']' then
          (Group(Vector(Lit("["), Group(p1), Lit("]"))), j + 1)
        else if c == ':' then
          try
            val (num, k) = numberClose(j + 1)
            (new Sched(None, p1, num), k)
          catch
            case _: Fail =>
              val (p2, k1) = prompt(j + 1)
              val k2 = expect(k1, ':')
              val (num, k3) = numberClose(k2)
              (new Sched(Some(p1), p2, num), k3)
        else if c == '|' then
          val opts = Vector.newBuilder[Vector[Node]]
          opts += p1
          var cur = j
          while cur < s.length && s.charAt(cur) == '|' do
            val (p, next) = prompt(cur + 1)
            opts += p
            cur = next
          (Alt(opts.result()), expect(cur, ']'))
        else throw new Fail

    def parse(): Vector[Node] =
      val out = Vector.newBuilder[Node]
      var i = 0
      while i < s.length do
        val (nodes, j) = prompt(i)
        out ++= nodes
        if j < s.length then
          val c = s.charAt(j)
          if "[]():".indexOf(c) >= 0 then
            out += Lit(c.toString)
            i = j + 1
          else throw new Fail
        else i = j
      out.result()

  private def walk(nodes: Vector[Node], fn: Node => Unit): Unit =
    nodes.foreach {
      case Group(items) => walk(items, fn)
      case n: Sched =>
        fn(n)
        n.before.foreach(walk(_, fn))
        walk(n.after, fn)
      case n @ Alt(opts) =>
        fn(n)
        opts.foreach(walk(_, fn))
      case n => fn(n)
    }

  private def render(nodes: Vector[Node], step: Int): String =
    val sb = new java.lang.StringBuilder
    def go(items: Vector[Node]): Unit = items.foreach {
      case Group(sub) => go(sub)
      case Lit(t) => sb.append(t)
      case n: Sched =>
        if step <= n.step then n.before.foreach(go)
        else go(n.after)
      case Alt(opts) =>
        go(opts((step - 1) % opts.size))
    }
    go(nodes)
    sb.toString

  def schedule(prompt: String, steps: Int, hiresSteps: Option[Int] = None): Vector[(Int, String)] =
    val (intOffset, fltOffset, total) = hiresSteps match
      case None => (0, 0.0, steps)
      case Some(hs) => (steps, 1.0, hs)
    val tree =
      try new Parser(prompt).parse()
      catch case _: Fail => return Vector((total, prompt))
    val endSteps = scala.collection.mutable.TreeSet[Int](total)
    walk(
      tree,
      {
        case n: Sched =>
          val v0 = n.when.toDouble
          val v = if n.when.contains('.') then (v0 - fltOffset) * total else (v0 - intOffset)
          n.step = math.min(total, v.toInt)
          if n.step >= 1 then endSteps += n.step
        case _: Alt =>
          (1 to total).foreach(endSteps += _)
        case _ => ()
      }
    )
    endSteps.toVector.map(t => (t, render(tree, t)))

  def hasSchedule(prompt: String): Boolean =
    prompt.contains('[') && schedule(prompt, 10).size > 1

  def textAt(sched: Vector[(Int, String)], step: Int): String =
    sched.find(_._1 >= step).map(_._2).getOrElse(sched.last._2)
