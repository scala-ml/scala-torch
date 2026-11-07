package scalatorch.ffm

class ProcessMemorySuite extends munit.FunSuite:
  test("process memory reports the current process working set") {
    val memory=ProcessMemory.current()
    if sys.props("os.name").toLowerCase(java.util.Locale.ROOT).startsWith("windows") then
      assert(memory.nonEmpty,"Windows process memory must be available")
      val info=memory.get
      assert(info.workingSetBytes>0)
      assert(info.peakWorkingSetBytes>=info.workingSetBytes)
      assert(info.privateBytes>0)
      val pid=ProcessHandle.current().pid()
      val proc=new ProcessBuilder("powershell.exe","-NoProfile","-Command",s"(Get-Process -Id $pid).WorkingSet64").redirectErrorStream(true).start()
      val output=new String(proc.getInputStream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8).trim
      assertEquals(proc.waitFor(),0)
      val observed=output.toLong
      assert(math.abs(info.workingSetBytes-observed)<128L*1024*1024,s"native=${info.workingSetBytes} os=$observed")
      for _ <- 1 to 500 do
        val next=ProcessMemory.current()
        assert(next.exists(_.workingSetBytes>0),"repeated sampling must remain available")
  }
