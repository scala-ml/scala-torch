# NUI Model Keyword reference data

`reference.jsonl` contains 144 static planner cases from NUI commit
`06454bff0adcc211ca8d50ae09467b7d9557ee13`. Source `python/sdfx/model_keyword.py`
SHA256: `da98493ec97b056c5753244c515f87c909837a1db52ddd132d93576e6e65f269`.
Fixture SHA256: `9626ea14111f022ddcdc13ffbe29ea6d0089b38e2e9df8f23decd6087e01f306`.

Regenerate using `python reference.py <NUI-checkout>`. Only fixture regeneration
uses Python. The Scala tests read the committed data without Python or NUI installed.
The generator calls NUI's existing `plan_with_preview` and `IterationState.reserve`.
Empty/`None` embedding selection is normalized as in NUI's `KeywordPlanner`.
Multipliers are serialized as strings to retain negative zero through JSON parsing.

Coverage includes joined, random and iterative choices, both orders, all four
placements, absent/empty/single/multiple keyword entries, optional embedding and
LoRA names, seed 42 and Long.MinValue. Additional tests check other signed seeds,
2,000 CPython `randrange(1000)` draws and selection/iteration boundary cases.

This verifies prompt planning only. Mapping lookup/editing, checkpoint directory
identity, worker hook integration, and actual LoRA/textual-inversion inference
are separate work. `ModelKeyword.render` produces LoRA tag text; it does not load
an adapter. `KeywordRandom` is for keyword choice, not Torch latent noise.

MT19937 provenance and redistribution notice ship in
`models/src/main/resources/META-INF/licenses/keyword-random.txt`.
