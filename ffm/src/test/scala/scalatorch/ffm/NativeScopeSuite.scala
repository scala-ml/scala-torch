package scalatorch.ffm

import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.JAVA_INT

class NativeScopeSuite extends munit.FunSuite:
  test("native memory is accessible only while scope is open") {
    val scope = new NativeScope()
    val memory = scope.allocate(4, 4)
    memory.set(JAVA_INT, 0L, 42)
    assertEquals(memory.get(JAVA_INT, 0L), 42)
    scope.close()
    scope.close()
    intercept[IllegalStateException](memory.get(JAVA_INT, 0L))
    intercept[IllegalStateException](scope.allocate(4, 4))
  }
  test("use closes memory when body throws") {
    var escaped: MemorySegment = MemorySegment.NULL
    intercept[IllegalArgumentException] {
      NativeScope.use { scope =>
        escaped = scope.allocate(4, 4)
        throw IllegalArgumentException("body failed")
      }
    }
    intercept[IllegalStateException](escaped.get(JAVA_INT, 0L))
  }
