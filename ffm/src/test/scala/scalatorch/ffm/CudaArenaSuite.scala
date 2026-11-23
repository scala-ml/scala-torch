package scalatorch.ffm

class CudaArenaSuite extends munit.FunSuite:
  /** A fake device: every chunk gets its own address range, far from the others. */
  private final class Device:
    var next=1L<<20
    val chunks=scala.collection.mutable.LinkedHashMap.empty[Long,Long]
    var rooms=Vector.empty[Long]
    val raw: Long => Long = bytes => { val a=next;next+=bytes+(1L<<20);chunks(a)=bytes;a }
    val rawFree: (Long,Long) => Unit = (address,bytes) => { assertEquals(chunks.remove(address),Some(bytes)) }
    val room: Long => Unit = bytes => rooms:+=bytes
    def total: Long = chunks.values.sum

  test("blocks never overlap, stay inside their chunk and merge again when freed") {
    val device=new Device
    val arena=new CudaArena(1L<<20)
    val random=new scala.util.Random(7)
    val live=scala.collection.mutable.LinkedHashMap.empty[Long,Long]
    for round <- 0 until 4000 do
      if live.nonEmpty && (random.nextInt(100)<48 || live.size>60) then
        val address=live.keys.toVector(random.nextInt(live.size))
        arena.release(address);live.remove(address)
      else
        val bytes=1L+random.nextInt(if random.nextInt(10)==0 then 900000 else 60000)
        val address=arena.allocate(bytes,device.room,device.raw,device.rawFree)
        assert(arena.blockBytes(address)>=bytes)
        assertEquals(address%256,0L)
        val end=address+arena.blockBytes(address)
        assert(device.chunks.exists((start,size) => address>=start && end<=start+size),"block outside a chunk")
        live.foreach { (other,size) => assert(end<=other || other+arena.blockBytes(other)<=address,s"overlap in round $round") }
        live(address)=bytes
      assertEquals(arena.liveBytes,live.keys.toVector.map(arena.blockBytes).sum)
      assertEquals(arena.ownedBytes,device.total)
    live.keys.toVector.foreach(arena.release)
    assertEquals(arena.liveBytes,0L)
    val owned=arena.ownedBytes
    assert(owned>0)
    assertEquals(arena.trim(device.rawFree),owned)
    assertEquals(device.total,0L)
    assertEquals(arena.ownedBytes,0L)
  }

  test("a freed block is reused for a smaller request and a large request gets a chunk of its own") {
    val device=new Device
    val arena=new CudaArena(1L<<20)
    val a=arena.allocate(300000,device.room,device.raw,device.rawFree)
    assertEquals(device.chunks.size,1)
    arena.release(a)
    val b=arena.allocate(1000,device.room,device.raw,device.rawFree)
    assertEquals(b,a)
    val big=arena.allocate(3L<<20,device.room,device.raw,device.rawFree)
    assertEquals(device.chunks(big),3L<<20)
    assertEquals(device.chunks.size,2)
    assertEquals(device.rooms,Vector(1L<<20,3L<<20))
  }

  test("empty chunks go back before the arena grows, and trim keeps chunks that are in use") {
    val device=new Device
    val arena=new CudaArena(1L<<20)
    val first=arena.allocate(2L<<20,device.room,device.raw,device.rawFree)
    val small=arena.allocate(1000,device.room,device.raw,device.rawFree)
    arena.release(first)
    // 4 MiB does not fit the freed 2 MiB chunk and the empty chunks are smaller than the request: nothing is returned yet.
    val second=arena.allocate(4L<<20,device.room,device.raw,device.rawFree)
    assertEquals(device.chunks.size,3)
    arena.release(second)
    // Now 6 MiB of chunks are empty: they are returned before an 5 MiB chunk is requested.
    val third=arena.allocate(5L<<20,device.room,device.raw,device.rawFree)
    assertEquals(device.chunks.keySet.size,2)
    assertEquals(device.chunks(third),5L<<20)
    assertEquals(arena.trim(device.rawFree),0L)
    arena.release(third);arena.release(small)
    assertEquals(arena.trim(device.rawFree),(5L<<20)+(1L<<20))
    assertEquals(device.total,0L)
  }

  test("a full chunk that the device refuses falls back to the exact size, and releaseAll forgets everything") {
    val device=new Device
    val arena=new CudaArena(1L<<20)
    var refused=0
    val stingy: Long => Long = bytes => if bytes==(1L<<20) then { refused+=1;throw IllegalStateException("no room") } else device.raw(bytes)
    val a=arena.allocate(1000,device.room,stingy,device.rawFree)
    assertEquals(refused,1)
    assertEquals(device.chunks(a),1024L)
    intercept[IllegalArgumentException](arena.release(a+256))
    arena.releaseAll(device.rawFree)
    assertEquals(device.total,0L)
    assertEquals(arena.ownedBytes,0L)
    assertEquals(arena.liveBytes,0L)
  }
