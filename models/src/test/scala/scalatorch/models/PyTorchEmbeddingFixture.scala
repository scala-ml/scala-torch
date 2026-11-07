package scalatorch.models

import java.io.{ByteArrayOutputStream,DataOutputStream}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import java.util.zip.{ZipEntry,ZipOutputStream}

/** Tiny torch.save-compatible fixtures for the allowlisted embedding archive subset. */
private[models] object PyTorchEmbeddingFixture:
  private final case class Item(key: String,values: Vector[Float],shape: Vector[Int])
  private def pickle(items: Vector[Item],layout: String): (Array[Byte],Map[String,Array[Byte]]) =
    val bytes=new ByteArrayOutputStream
    val out=new DataOutputStream(bytes)
    def op(value: Int): Unit=out.writeByte(value)
    def raw(value: String): Unit=out.write(value.getBytes(UTF_8))
    def global(module: String,name: String): Unit={ op('c');raw(module);op('\n');raw(name);op('\n') }
    def unicode(value: String): Unit=
      val data=value.getBytes(UTF_8);op('X');out.writeInt(Integer.reverseBytes(data.length));out.write(data)
    def int(value: Int): Unit={ op('J');out.writeInt(Integer.reverseBytes(value)) }
    def tuple(values: Vector[Int]): Unit={ op('(');values.foreach(int);op('t') }
    def tensor(item: Item,storageKey: String): Unit=
      global("torch._utils","_rebuild_tensor_v2");op('(');op('(')
      unicode("storage");global("torch","FloatStorage");unicode(storageKey);unicode("cpu");int(item.values.length)
      op('t');op('Q');int(0);tuple(item.shape)
      tuple(item.shape.indices.map(i=>item.shape.drop(i+1).product).toVector)
      op(0x89);global("collections","OrderedDict");op(')');op('R');op('t');op('R')
    op(0x80);op(2)
    layout match
      case "tensor" => tensor(items.head,"0");op('.')
      case "emb_params" => op('}');unicode("emb_params");tensor(items.head,"0");op('s');op('.')
      case "string_to_param" =>
        op('}');unicode("string_to_param");op('}');unicode("*");tensor(items.head,"0");op('s');op('s')
        unicode("name");unicode("fixture");op('s');unicode("step");int(123);op('s');op('.')
      case "sdxl" =>
        op('}')
        items.zipWithIndex.foreach { (item,index) => unicode(item.key);tensor(item,index.toString);op('s') }
        unicode("step");int(123);op('s');op('.')
      case other => throw IllegalArgumentException(s"unknown fixture layout: $other")
    out.flush()
    val pickle=bytes.toByteArray
    val storage=items.zipWithIndex.map { (item,index) =>
      val raw=ByteBuffer.allocate(item.values.size*4).order(ByteOrder.LITTLE_ENDIAN)
      item.values.foreach(raw.putFloat)
      index.toString->raw.array()
    }.toMap
    pickle->storage

  def write(path: Path,layout: String,items: (String,Vector[Float],Vector[Int])*): Path=
    val decoded=items.toVector.map((key,values,shape)=>Item(key,values,shape))
    val (pickleBytes,storages)=pickle(decoded,layout)
    val zip=new ZipOutputStream(Files.newOutputStream(path))
    try
      def entry(name: String,data: Array[Byte]): Unit={ zip.putNextEntry(new ZipEntry(name));zip.write(data);zip.closeEntry() }
      entry("archive/data.pkl",pickleBytes)
      storages.foreach((key,data)=>entry(s"archive/data/$key",data))
      entry("archive/version","3\n".getBytes(UTF_8))
    finally zip.close()
    path
