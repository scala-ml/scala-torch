package scalatorch.ffm

import java.lang.foreign.{Arena,FunctionDescriptor,Linker,MemorySegment,SymbolLookup,ValueLayout}
import java.util.Locale
import scala.util.control.NonFatal

/** Process-wide host memory; values are not JVM heap or GPU memory. */
object ProcessMemory:
  final case class Info(workingSetBytes: Long,peakWorkingSetBytes: Long,privateBytes: Long)
  private lazy val windowsApi =
    val linker=Linker.nativeLinker()
    val dll=SymbolLookup.libraryLookup("kernel32",Arena.global())
    val current=linker.downcallHandle(dll.find("GetCurrentProcess").orElseThrow(),
      FunctionDescriptor.of(ValueLayout.ADDRESS))
    val getInfo=linker.downcallHandle(dll.find("K32GetProcessMemoryInfo").orElseThrow(),
      FunctionDescriptor.of(ValueLayout.JAVA_INT,ValueLayout.ADDRESS,ValueLayout.ADDRESS,ValueLayout.JAVA_INT))
    (current,getInfo)
  def current(): Option[Info] =
    if !sys.props.getOrElse("os.name","").toLowerCase(Locale.ROOT).startsWith("windows") ||
      !Set("amd64","x86_64")(sys.props.getOrElse("os.arch","").toLowerCase(Locale.ROOT)) then None
    else try
      val (current,getInfo)=windowsApi
      val arena=Arena.ofConfined()
      try
        // PROCESS_MEMORY_COUNTERS_EX on Windows x64: two DWORDs then nine SIZE_Ts.
        val size=80
        val buffer=arena.allocate(size,8)
        buffer.set(ValueLayout.JAVA_INT,0,size)
        val handle=current.invokeWithArguments().asInstanceOf[MemorySegment]
        val ok=getInfo.invokeWithArguments(handle,buffer,Int.box(size)).asInstanceOf[java.lang.Integer].intValue()
        if ok==0 then None else
          val peak=buffer.get(ValueLayout.JAVA_LONG,8)
          val working=buffer.get(ValueLayout.JAVA_LONG,16)
          val privateBytes=buffer.get(ValueLayout.JAVA_LONG,72)
          if working<=0 || peak<working || privateBytes<0 then None else Some(Info(working,peak,privateBytes))
      finally arena.close()
    catch case NonFatal(_) | _: LinkageError => None
