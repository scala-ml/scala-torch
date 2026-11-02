# 再利用と置換境界の一次監査

確認日: 2026-11-02。GitHubのtreeを固定revisionで取得し、下記ソースを読んだ。これは全コード監査・実行検証ではない。

## numscala

revision: `d520971cfdc599a111df79616dc1b781066c5351`

- [MatrixF32.scala](https://github.com/scala-ml/numscala/blob/d520971cfdc599a111df79616dc1b781066c5351/src/main/scala/com/github/kmizu/numscala/cpu/MatrixF32.scala)
- [F32Kernels.scala](https://github.com/scala-ml/numscala/blob/d520971cfdc599a111df79616dc1b781066c5351/src/main/scala/com/github/kmizu/numscala/cpu/F32Kernels.scala)

`MatrixF32(data: Array[Float], offset: Int, rows: Int, cols: Int, rowStride: Int)`はbufferを所有・copyしないdescriptor。行内strideは1であり、任意のTensor strideを直接表現できるわけではない。通常の行列と転置利用はphysical shapeと`Transpose.No/Yes`で接続できる。broadcastのstride 0や一般的なsliceは適合判定の後でpackingする必要がある。

`F32Kernels.gemmInto(a, transA, b, transB, c, alpha, beta, workspace): Unit` はcaller-owned outputへ書く。Cと入力の重複を拒否する。`beta == 0`はCを読まず、K=0またはalpha=0はA/Bを読まない。adapterのテストに、この読み取り省略とNaNを含める。workspaceはworkerごとに管理し、同じものを並行共有しない。TT積のworkspace要求も見積る。

gatherRows/scatterAddRowsもtraitにあり、Embeddingのforward/backward候補になる。Int64のtoken IDをIntへ無検証に変換せず、範囲検査してから接続する。

## Stable Diffusion NUI

revision: `1f41743b3d6b2f9e76c2d6346e2e98646b921a35`

- [Protocol.scala](https://github.com/kmizu/stablediffusion-nui/blob/1f41743b3d6b2f9e76c2d6346e2e98646b921a35/core/src/main/scala/sdfx/core/protocol/Protocol.scala)
- [EngineProcess.scala](https://github.com/kmizu/stablediffusion-nui/blob/1f41743b3d6b2f9e76c2d6346e2e98646b921a35/engine/src/main/scala/sdfx/engine/EngineProcess.scala)

protocolには生成以外にもmetadata、embedding merge、system benchmark、VRAM estimator、temporal lab等がある。READMEの生成機能だけを移植して完了にできない。requestのid付きresult/error/eventと最初のhelloを維持する。

cancelには単純なinterruptだけでなく`interruption_protection`と`restart_authorization`がある。`InterruptionDecision.validAdmission`はtooLateとaccepted/blocked/affectedの整合性を定める。Scala workerでもこの契約を保ち、拒否されたcancelがnative処理を中断しないよう結合試験を作る。

`GenerationResult`は画像以外にwarnings、intermediateArtifacts、resourceHashes、interrupted、auxiliaryImages等を持つ。これらを捨てるとUI/MCPの機能が欠落する。`EngineHello`にpython/torch/diffusersのフィールドがあるため、Scala workerのbackend表現と既存readerとの互換性も移行対象になる。

## 次の監査

NDArray storageの公開境界、Workspace実装、EngineClient/Supervisor/Generator/PythonEnv、Python server dispatch、全pluginと設定を読む。全request/feature台帳を埋めるまではM0監査完了にしない。今回の知見はadapter設計とNUI結合試験の要件として扱う。

A subsequent [NUI source inventory](nui-migration-audit.md) pins revision8823aa413a3b3a1db682456357b15ee681387554 and enumerates the dispatch table, standard plugins, parameter fields and Python modules. This earlier revision remains historical evidence; neither audit claims complete behavioral verification.
