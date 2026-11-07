package scalatorch.io

import scalatorch.*
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8

/** Bounded in-memory safetensors interchange for a single tensor named tensor. */
object SafeTensorMemory:
  def writeFloat(tensor: Tensor[Float]): Array[Byte] =
    require(tensor.device==Device.Cpu && tensor.dtype==DType.Float32,"expected CPU Float32 tensor")
    val values=tensor.toFloatArray
    val length=Math.multiplyExact(values.length,4)
    val json=s"""{"tensor":{"dtype":"F32","shape":[${tensor.shape.dims.mkString(",")}],"data_offsets":[0,$length]}}"""
    val header=json.getBytes(UTF_8)
    val padded=(header.length+7)& ~7
    val result=ByteBuffer.allocate(Math.addExact(Math.addExact(8,padded),length)).order(ByteOrder.LITTLE_ENDIAN)
    result.putLong(padded.toLong).put(header)
    while result.position()<8+padded do result.put(32.toByte)
    values.foreach(result.putFloat)
    result.array()

  def readFloat(bytes: Array[Byte],limits: ReadLimits = ReadLimits()): Tensor[Float] =
    require(bytes!=null && bytes.length>=8,"truncated safetensors")
    val buffer=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val headerLength=buffer.getLong()
    require(headerLength>=2 && headerLength<=limits.maxHeaderBytes && headerLength<=bytes.length-8,"invalid safetensors header length")
    val start=8+headerLength.toInt
    val header=SafeTensorsHeader.parse(java.util.Arrays.copyOfRange(bytes,8,start),bytes.length-start,limits)
    require(header.tensors.keySet==Set("tensor"),"expected exactly one tensor")
    val info=header.tensors("tensor")
    require(Set(SafeDType.F16,SafeDType.BF16,SafeDType.F32,SafeDType.F64)(info.dtype),"expected floating tensor")
    require(info.shape.numel<=Int.MaxValue && info.shape.numel*4L<=limits.maxTensorBytes,"tensor exceeds allocation limit")
    buffer.position(start+info.begin.toInt)
    val values=Array.fill(info.shape.numel.toInt) {
      info.dtype match
        case SafeDType.F16 => java.lang.Float.float16ToFloat(buffer.getShort())
        case SafeDType.BF16 => java.lang.Float.intBitsToFloat((buffer.getShort().toInt & 0xffff)<<16)
        case SafeDType.F32 => buffer.getFloat()
        case SafeDType.F64 => buffer.getDouble().toFloat
        case _ => throw IllegalStateException("invalid dtype")
    }
    Tensor.fromFloats(values,info.shape)
