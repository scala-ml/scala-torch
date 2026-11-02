# scala-torch

Scala 3でモデルを書き、学習し、実行するための深層学習基盤。Java 25を前提に、PythonなしでLLMと画像生成を扱うことを目指す。

現在は計画段階。Tensor演算や推論が実装済みという意味ではない。

- [設計・ロードマップ](docs/superpowers/specs/2026-11-02-scala-torch-design.md)
- [最初の実装計画](docs/superpowers/plans/2026-11-02-tensor-foundation.md)
- [既存コードの再利用・置換境界の調査](docs/research/reuse-audit.md)

PyTorchの全API互換ではなく、Scalaでモデルを簡潔に記述・変更できることと、実モデルをPythonなしで動かせることを成功条件とする。
