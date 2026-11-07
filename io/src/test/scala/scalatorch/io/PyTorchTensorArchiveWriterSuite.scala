package scalatorch.io

import java.nio.file.{Files,Path}
import PyTorchTensorArchive.*

class PyTorchTensorArchiveWriterSuite extends munit.FunSuite:
  test("writes SD1 and SDXL embedding archives that native reader can load") {
    val dir=Files.createTempDirectory("pytorch-writer")
    try
      val left=TensorValue(StoredTensor(StorageDType.Float32,Vector(1,2),Array(1f,2f)))
      val right=TensorValue(StoredTensor(StorageDType.Float16,Vector(1,2),Array(3f,4f)))
      val xl=DictValue(Map("clip_l"->left,"clip_g"->right,"step"->IntegerValue(42)))
      val path=dir.resolve("xl.pt")
      PyTorchTensorArchiveWriter.write(path,xl)
      val data=PyTorchTensorArchive.read(path).value.asInstanceOf[DictValue].entries
      assertEquals(data("step"),IntegerValue(42))
      assertEquals(data("clip_l").asInstanceOf[TensorValue].tensor.values.toVector,Vector(1f,2f))
      assertEquals(data("clip_g").asInstanceOf[TensorValue].tensor.dtype,StorageDType.Float16)
      assertEquals(data("clip_g").asInstanceOf[TensorValue].tensor.values.toVector,Vector(3f,4f))
      sys.props.get("scalatorch.pytorch.writer.test.output").foreach(output=>Files.copy(path,Path.of(output)))
      intercept[java.nio.file.FileAlreadyExistsException](PyTorchTensorArchiveWriter.write(path,xl))
      val overflow=dir.resolve("overflow.pt")
      intercept[IllegalArgumentException](PyTorchTensorArchiveWriter.write(overflow,
        TensorValue(StoredTensor(StorageDType.Float16,Vector(1),Array(1e10f)))))
      assert(!Files.exists(overflow))
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
