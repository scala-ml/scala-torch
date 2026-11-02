package scalatorch

class EnvironmentSuite extends munit.FunSuite:
  test("runs on Java 25") {
    assertEquals(Runtime.version().feature(), 25)
  }
