# scala-torch 設計・ロードマップ

作成日: 2026-11-02。状態: ユーザーの `approve` により承認済み。実装進行中。

## 目的と成功条件

Scalaを深層学習の実装言語にする。Java向けAPIをScalaで包むだけにせず、Scala 3の型、enum、extension、inline、derivationを使い、モデル・学習ループ・推論パイプラインを簡潔に記述できるようにする。

最終的に、(1)公開重みからLLMをロードして対話・バッチ推論する、(2)小型モデルの学習とLoRA微調整をScalaで行う、(3)Stable Diffusion NUIのPythonエンジンをScala実装に置換する。PyTorchの全API・Python構文・pickle・分散学習の完全互換は要求しない。既存モデルの入出力仕様と必要な演算の数値的意味は互換性の対象とする。

「Pythonなし」は利用者のインストール、モデル読込、推論、学習、配布時にPythonプロセス・CPython埋込み・pipを必要としないこと。開発時の比較用PyTorchは隔離された任意の参照環境だけに許す。モデルごとのPython変換を必須にしない。

## 現状の根拠

- この作業ディレクトリは空で、Gitリポジトリもビルドも未作成だった。
- `java -version` は OpenJDK 25.0.1 (build 25.0.1+8-27)。追加調査でGTX 1650 SUPER（VRAM 4096 MiB）、GCC、sbt配置を確認した。[環境記録](../../research/baseline.md)を参照。CUDA演算・Scalaビルドの成功はまだ未検証。
- [numscala README](https://github.com/scala-ml/numscala/blob/main/README.md) / [build.sbt](https://github.com/scala-ml/numscala/blob/main/build.sbt): Scala 3.3.7、JDK 17基盤、NDArray、strides/views、Float32 CPU kernels、別artifactのVector25がある。READMEの0.5.0表記は取得時点の記述であり、公開artifactの取得は実装時に検証する。float16/bfloat16やGPU storageはこの調査では再利用可能と確認できていない。
- [Stable Diffusion NUI README](https://github.com/kmizu/stablediffusion-nui/blob/main/README.md): Scala/JavaFXのUIとMCPがPython diffusersをJSON Linesで呼ぶ。置換境界は `core/src/main/scala/sdfx/core/protocol/Protocol.scala` とengine。本文を調査したのはREADMEとnumscalaのbuildで、両プロジェクトの実装監査は最初の工程に含める。
- [JDK 25 FFM公式文書](https://docs.oracle.com/en/java/javase/25/core/foreign-function-and-memory-api.html): `java.lang.foreign` からネイティブ関数・メモリを扱える。FFMがC++ ABIを直接解決したり、CUDAカーネルを自動生成したりするわけではない。

## 方式の比較と推奨

| 方式 | 利点 | 代償 |
|---|---|---|
| LibTorch主体 | 演算・autograd・GPU資産の再利用 | C++ APIとの境界にC ABIのshim等が必要。配布とABI追従が重くScala側の実装主導権が小さい |
| ScalaのTensor/autograd + FFMの演算backend | モデルや微分がScala主体。C ABIのBLAS/CUDAはJNIなしで呼べる | 演算・微分・デバイス管理を段階的に実装する必要がある |
| 全演算をScalaから自作 | 外部依存を最小化 | GPU kernels、融合、性能検証まで抱え、実モデル到達が遅い |

推奨は2番目。これは作者の依頼から導いた提案で、ユーザーが選択済みとは扱わない。LibTorchや専用推論エンジンを任意backendとして後から比較できるが、それだけでScalaの学習基盤の完成とはしない。

## 制約と構成

- Scala 3.3.8 LTS, Java 25, and sbt 2.x are required by the user. The build pins sbt 2.0.10. Earlier references to Scala 3.3.7 describe the inspected numscala build, not this project's selected compiler.
- Python不要。Scalaが公開APIとモデル実装を所有する。JNI/JavaCPPを標準の境界にしない。
- `core`: Tensor、shape/stride、dtype、storage所有権、device、dispatch。`cpu`: numscala kernelsとの接続。`ffm`: symbol/layout/error/lifetime。`cuda`: device memory、stream、event、演算。`autograd`、`nn`、`optim`、`io`、`models`、`examples`をその上に分離する。
- numscalaはCPUの演算資産と相互変換先。NDArrayをGPU Tensorそのものに見立てない。まず明示的copyで接続し、zero-copyは寿命とaliasを検証してから追加する。NumPyとPyTorchの型昇格規則を混同しない。
- Tensorはdtypeとdeviceを保持し、shapeはまず実行時検証。巨大な型レベルshape計算を利用者に強制しない。opaque type/inlineでboxingを減らし、derivationでModuleのparameter登録とstate_dictを簡略化する。macroは測定または重複削減の根拠がある箇所だけ使う。

## Tensorと微分の契約

初期CPUはFloat32、Float64、Int64、Bool。FP16/BF16はGPU混合精度までに追加する。shape積・offset計算はLongでoverflowを検査する。scalarはrank 0、空次元を認める。viewはstorageを共有し、reshapeでcopyが必要なら明示する。device転送を演算が黙って行わない。

CPU heapとnative/device storageを分離し、後者は明示的scopeで閉じる。viewが残るstorageの解放、非同期CUDA完了前の解放を防ぐ。use-after-closeは例外、native crashにしない。初期APIはin-place演算を制限し、autogradのversion counterとalias規則が揃うまで拡大しない。

eager reverse-modeから始める。broadcast backwardは元shapeへsum、共有parameterの勾配は加算。noGrad、detach、zeroGrad、gradient accumulationを定義する。非対応の微分は明示エラー。高階微分、graph compiler、分散は後続。

## ネイティブと性能

C ABIを持つBLASおよびCUDA Driver/Runtime、cuBLAS系を候補に、公式headers/docsを確認してFFM bindingを実装する。ライブラリ別version、OS/arch、symbol availabilityを記録する。cuDNN等や独自kernelが必要な演算はABI・ビルド工程を個別設計する。「グルー不要」はJNI不要という利点であり、binding、error検査、kernel準備、配布を不要とはしない。

FFM spikeはalloc/copy/GEMM/freeを実GPUで通す。streamとeventを所有し、失敗時も一度だけ解放。初期はLinux x86_64 CPUを必須gate、CUDAは実機gate、Windows x64はNUI移行前の必須gate。GPU未搭載時のskipをGPU成功と報告しない。macOS CPUは後続、Metalは別設計。

手元の初期実機条件は4GB VRAM。モデルロード前に重み・activation・KV cache・workspaceの見積りを提示し、GPU常駐できないモデルはCPUまたは明示的offloadを選べるようにする。保存dtypeとGPUでの演算対応を分け、device capabilityを検出して非対応の混合精度を拒否する。

JMHでwarmup、threads、shape、dtype、JDK、CPU/GPU、転送有無を記録。LLMではtokens/s・初回token時間・KV cacheと総メモリ、SDでは画像時間・peak VRAMを測る。基準値を取ってから性能目標を合意し、根拠なくPyTorch同等と宣言しない。

## 段階と完了条件

各段階は前段の成果を利用する。小さなデモを全体完成にすり替えない。

| 段階 | 成果 | 必須の証拠 |
|---|---|---|
| M0 調査・実現性 | numscala storage/kernel監査、NUI protocol/feature台帳、環境・依存固定、FFM CPU/GPU spike | commit SHA付き調査、JDK25 build、CPU参照GEMM、実GPUのcopy/GEMM結果または明記された未実施gate |
| M1 Tensor | CPU storage/view/broadcast/reduce/matmul、numscala接続 | 非連続・空・scalar・overflow・alias試験、数値比較、割当計測 |
| M2 学習基盤 | reverse autograd、Module/Linear/Embedding/norm/activation、SGD/AdamW、state_dict | finite difference、重み共有・broadcast勾配、小型MLPのloss低下、save/load再開 |
| M3 LLM CPU | safetensorsとconfig/tokenizer直読、RMSNorm/RoPE/GQA/causal attention/KV cache、sampling | 固定revisionの小型Llama系1モデルでtokenizer一致、prefill/decode logits比較、Pythonなしoffline生成。モデル名・hash・許容誤差はfixture登録時に固定 |
| M4 GPUと微調整 | FP16/BF16、CUDA dispatch、attention改善、混合精度、LoRA | CPU/GPU誤差、OOM/cancel/解放、実モデル生成、LoRA loss低下・保存再読込、性能とVRAM報告 |
| M5 Diffusion | CLIP tokenizer/text encoder、UNet、VAE、conv2d/group norm/upsample、scheduler/CFG、SD1.5からSDXLへ | 部品別参照比較、同一latent入力のdenoise比較、固定条件の実画像。画像のbyte一致を乱数実装差のある比較に要求しない |
| M6 NUI完全移行 | Scala engineへ置換、Python setupと起動経路除去、既存機能移植 | Windows配布版をPythonのない環境で起動しUI/MCPの実生成、cancel、preview、再起動、offlineを試験。全機能台帳が移植・検証済み |

M3はinferenceだけでなく、後続M4の微調整につながる汎用演算を使う。量子化(Int8/Int4、GGUF等)はモデルサイズの実測後に追加し、まず非量子化の意味を検証する。tokenizerはBPEだけで全モデル対応としない。対応familyごとにnormalization、特殊token、chat template、stop条件を固定する。

## NUIのPython除去を判定する台帳

M0でREADMEと実コードからrequest/event/設定/pluginの行単位台帳を作り、各行にScala実装箇所と結合テストを対応させる。最低限txt2img、img2img、inpaint、hires、LoRA、VAE、clip skip、ControlNetと前処理、prompt強調/chunk、batch/queue/seed、preview、upscaler、caption/WD、face restore、動画系、全標準Script/AlwaysOnを含める。Python Custom codeはScala拡張APIへの移行案を別途示す必要があり、黙って削除して「完全除去完了」としない。

protocolを維持したJVM子プロセスから移行し、in-process化は後で判断できる。Python設定、venv導入、Pythonによる第三者notice生成まで置換し、実行・配布・保守で残る依存を確認する。モデル・tokenizer・native libraryの取得とcacheはScalaから行い、hashとoffline動作を検証する。

## 品質・配布

safetensorsを優先し、header長、dtype、shape積、offset範囲・重複、truncationを検証する。pickleは読み込まない。モデルrevisionとSHA256、参照環境、絶対/相対誤差をfixture manifestに保存し、通常テストは小さい固定データでPythonなしに実行する。

PR gateはJDK25 CPU testsとcompile例。GPU gateは実機で独立表示し、release前に必須とする。native依存はOS/arch別、利用可能backendを明示し、失敗を黙ってCPUへfallbackしない。coreのライセンスは候補BSD-3-Clauseとして別途決定する。NUIのAGPL由来コードはcoreへ無検討にコピーしない。モデル・native配布条件とnoticeを個別記録する。

## 次の意思決定

この提案のbackend方針をレビューした後、M0/M1を実装する。実装前に決めるのはGPU実機の利用方法、最初の公開モデルと利用可能な重み、NUI台帳の移行順。GPUやモデルが手元になくてもCPU基盤・loaderの開発は進められるが、その状態でGPU/実モデルgateを完了にはしない。
