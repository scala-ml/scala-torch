# ローカル環境の確認

確認日: 2026-11-02。読み取りによる環境調査。ビルド・FFM呼出しの成功を示すものではない。

| 項目 | 実測 |
|---|---|
| OS/arch | Linux x86_64（WSLのGPU管理ツールを使用） |
| Java / javac | OpenJDK 25.0.1 / javac 25.0.1 |
| Java配置 | `/home/mizushima/.sdkman/candidates/java/current` |
| sbt配置 | SDKMANのcurrentは`/home/mizushima/.sdkman/candidates/sbt/2.0.10`を指す。プロジェクトでの起動・compiler互換性は未検証 |
| C compiler | GCC 15.2.0 |
| GPU | NVIDIA GeForce GTX 1650 SUPER |
| VRAM | 4096 MiB |
| NVIDIA driver | 610.88（nvidia-smi報告） |
| Git / build | このディレクトリにはまだ`.git`、`build.sbt`なし |

確認コマンド: `java -version`, `javac -version`, `uname -sm`, `command -v java javac sbt gcc nvidia-smi`, `gcc --version`, `readlink -f /home/mizushima/.sdkman/candidates/sbt/current`, `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`。

## 計画への影響

- 4GB VRAMを初期の実機検証条件に加える。重みだけでなくactivation、KV cache、workspace、allocator、UI利用分を見積もる。大きいモデルはCPU実行または利用者が選択する明示的offloadが必要。
- BF16/FP16の保存形式への対応と、そのGPUで高速に計算できることを区別する。実際のdevice capabilityとlibrary対応をprobeで取得し、対応しないdtypeを明示する。
- sbt launcherの配置versionからプロジェクトのsbt versionを推測しない。numscalaのbuildとの互換性を確認して`project/build.properties`を固定する。
- `nvidia-smi`の成功はCUDA Toolkit、cuBLAS、FFM bindingの成功を保証しない。native libraryごとにload・symbol・小演算を検証する。

The above records the initial environment discovery. Following user approval, the build now uses Scala 3.3.8 and sbt 2.0.10 on Java 25; see [implementation evidence](implementation-status.md) for current verified behavior. Git was initialized on `feat/tensor-foundation`.
