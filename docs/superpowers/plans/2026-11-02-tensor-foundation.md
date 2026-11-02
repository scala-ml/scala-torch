# Tensor Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Java 25上で、後続のautogradとCPU/CUDA backendを支えるScala Tensor基盤を作る。

**Architecture:** Tensorのmetadataとstorageを分離し、CPU参照実装で意味を固定する。numscalaはCPU kernel adapterから再利用し、FFMは独立spikeでABIと寿命を確認する。

**Tech Stack:** Scala 3（初期候補3.3.7）、Java 25、sbt、MUnit、numscala、FFM。

**Spec:** `docs/superpowers/specs/2026-11-02-scala-torch-design.md`

この計画はM0/M1を詳細化する。M2–M6は設計書の必須到達点として残り、それぞれ前段の計測結果から詳細計画を書く。設計は未承認の提案であり、この文書だけで実装開始済みとはしない。

## Global Constraints

- Scala 3、Java 25必須。
- Python不要。Scalaが公開APIとモデル実装を所有する。JNI/JavaCPPを標準の境界にしない。
- scalarはrank 0、空次元を認める。shape積・offset計算はLongでoverflowを検査する。
- device転送を演算が黙って行わない。
- GPU未搭載時のskipをGPU成功と報告しない。

## Review Focus

- shape積overflowと不正view: storageアクセス前に拒否（Task 2）。
- 空Tensorとrank 0: 要素数とbroadcastが一貫（Task 2/3）。
- transpose/sliceのalias: 元storageと正しく共有（Task 3）。
- dtype/device不一致: 暗黙転送や誤昇格をしない（Task 4）。
- native非同期処理と解放: 完了前に解放しない（Task 5）。

---

### Task 1: 根拠・環境を固定する

**Files:** Create `docs/research/baseline.md`, `docs/research/nui-features.csv`, `build.sbt`, `project/build.properties`, `.gitignore`, `core/src/test/scala/scalatorch/EnvironmentSuite.scala`。

**Interfaces:** 後続taskへJDK/compiler/sbt/numscalaの検証済みversionとCPU kernelの実signatureを渡す。

- [ ] numscalaとNUIのcommit SHAを記録し、storage、kernel、protocol、Python起動、全pluginを読む。台帳の列をfeature/source/request/event/Scala replacement/test/statusとし、未移植は未完了にする。
- [ ] `java -version`, `javac -version`, `sbt --version`, `nvidia-smi` の結果をbaselineへ保存する。存在しないtoolを「利用可能」としない。
- [ ] `EnvironmentSuite`に `Runtime.version().feature() == 25` をassertするテストを書き、build未作成の失敗を確認する。
- [ ] core/cpu/ffmのsbt moduleとMUnitを定義し、JDK25でScala候補とnumscala artifactを実際にcompileする。失敗時は対応compilerを調べ、選択理由を記録する。
- [ ] `sbt core/test`で環境テストPASSを確認し、検証済みversionを固定してコミットする。

### Task 2: ShapeとCPU storage

**Files:** Create `core/src/main/scala/scalatorch/{Shape,DType,Device,Storage,Tensor}.scala`, `core/src/test/scala/scalatorch/TensorStorageSuite.scala`。

**Interfaces:** `Shape.from(dims: Seq[Long]): Shape`, `Shape.numel: Long`; `Tensor.fromFloats(values: Array[Float], shape: Shape): Tensor[Float]`; `Tensor.shape: Shape`; `Tensor.toFloatArray: Array[Float]`（copy）。DType enumはFloat32/Float64/Int64/Bool、Device enumはCpu/Cuda(index: Int)。CPU以外の作成はbackend未実装エラー。

- [ ] テストを書く: `Shape.from(Seq()).numel == 1`、`Seq(2,0,3)`は0、負次元拒否、`Seq(Long.MaxValue,2)`overflow拒否。要素数とbuffer長不一致拒否、入力配列の変更がTensorへ漏れないことをassertする。
- [ ] `sbt 'core/testOnly scalatorch.TensorStorageSuite'`で未実装FAILを確認する。
- [ ] metadataとprivate CPU primitive storageを実装する。JVM配列のInt上限を超えるallocationは明確に拒否し、Long metadata対応と大容量allocation対応を区別する。
- [ ] 同commandでPASSを確認し、仕様とコードをコミットする。

### Task 3: View・broadcastの意味

**Files:** Modify `Tensor.scala`; Create `core/src/main/scala/scalatorch/Layout.scala`, `core/src/test/scala/scalatorch/TensorViewSuite.scala`。

**Interfaces:** `Tensor.transpose(dim0: Int, dim1: Int): Tensor[A]`, `Tensor.reshape(shape: Shape): Tensor[A]`（非連続なら拒否）, `Tensor.contiguous: Tensor[A]`（必要時copy）, `Tensor.broadcastTo(shape: Shape): Tensor[A]`。共有storageとoffset/strideは内部APIとして保持。

- [ ] テストを書く: 2x3のtransposeは3x2で値が正しくstorage共有、非連続reshapeは拒否、contiguous後は可能、scalar→2x3 broadcast、(2,1)→(2,3)、(0,1)→(0,3)、(2,3)→(2,4)拒否。内部テストでoffset範囲外を拒否する。
- [ ] `sbt 'core/testOnly scalatorch.TensorViewSuite'`でFAILを確認する。
- [ ] checked layout計算とviewを実装する。公開in-place操作は追加しない。broadcast stride 0と空viewを正しく扱う。
- [ ] 同commandと`core/test`でPASSを確認しコミットする。

### Task 4: CPU演算とnumscala adapter

**Files:** Create `core/src/main/scala/scalatorch/Ops.scala`, `cpu/src/main/scala/scalatorch/cpu/NumscalaBackend.scala`, `cpu/src/test/scala/scalatorch/cpu/BackendSuite.scala`。

**Interfaces:** 初期Float32対象の `Ops.add(a: Tensor[Float], b: Tensor[Float]): Tensor[Float]`, `Ops.sum(a: Tensor[Float], dims: Seq[Int], keepDim: Boolean): Tensor[Float]`, `Ops.matmul(a: Tensor[Float], b: Tensor[Float]): Tensor[Float]`。matmulはまずrank 2、他rankは明示エラー。adapterはTask 1で確認したCPU GEMMへ接続。

- [ ] broadcast add、transpose入力GEMM、空の縮約和0、既知の2x3×3x2積、shape不一致・device不一致拒否をテストする。GEMMは小fixtureに対し `atol=1e-5, rtol=1e-5`、NaNは別assertとする。
- [ ] `sbt 'cpu/testOnly scalatorch.cpu.BackendSuite'`でFAILを確認する。
- [ ] reference loopsとadapterを実装し、NumPyの昇格をTensorへそのまま持ち込まない。明示copyのnumscala相互変換を追加する場合はcopy/alias契約をテストする。
- [ ] 同commandと`core/test cpu/test`でPASSを確認する。JMH用moduleに連続/転置GEMMのbenchmarkを用意し、初回の割当・所要時間を保存してコミットする。

### Task 5: FFM実現性を独立に検証

**Files:** Create `ffm/src/main/scala/scalatorch/ffm/{NativeScope,BlasProbe,CudaProbe}.scala`, `ffm/src/test/scala/scalatorch/ffm/NativeScopeSuite.scala`, `docs/research/native-backends.md`。

**Interfaces:** `NativeScope.use[A](f: NativeScope => A): A`; probeのCLIはbackendのversion、device、入力shape、誤差、解放結果を出力し、失敗は非0で終了する。probeを公開Tensor GPU実装と称さない。

- [ ] CPUで二重closeの安全性、close後アクセス拒否、例外時の解放をテストし、未実装FAILを確認する。
- [ ] 対象native libraryの公式header/docsからsymbolとFunctionDescriptorを定義する。FFM native access設定をforkしたtest/runへ明示する。
- [ ] CPU native GEMMをScala参照結果と比較する。実GPUがあればalloc→host/device copy→GEMM→event完了待ち→copy back→freeを実行し、同じ小fixtureとの誤差を検証する。途中失敗も解放する。
- [ ] `sbt ffm/test`を実行し、probeの再現commandと結果を記録する。GPUがなければCUDA gateは未実施のまま残す。
- [ ] M1の契約・実測・残課題を設計書と照合し、M2 autogradの詳細計画を作成する。承認された範囲のコードと証拠をコミットする。

## 完了監査

- [ ] JDK25で`core/test cpu/test ffm/test`を実行し、結果・環境・commitを記録した。
- [ ] READMEのAPI例がcompileされ、実装済みと未実装が区別されている。
- [ ] M0/M1の全契約と台帳が埋まっている。未実施GPU gateは明記した。
- [ ] M2–M6を完了扱いにしていない。実モデル・NUI移行は別途検証が必要。
