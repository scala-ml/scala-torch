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

## Mapping references

`mapping_reference.py <NUI-checkout>` generates `similarity.jsonl` (160 cases)
and `csv.jsonl` (10 cases). These use Python 3.12.12 `difflib.SequenceMatcher`
and the same NUI source revision listed above. They cover asymmetric matching,
autojunk's 200-character threshold, repeated characters, non-BMP Unicode,
multiline/escaped/empty CSV fields, and Unicode whitespace stripping.
Python's redistribution terms ship as `META-INF/licenses/python.txt`.

`KeywordMappings` rereads files on each lookup; it does not edit them or create
lock files. Reads compare size, modification time and file identity before/after
reading and fail if a concurrent replacement is observed. This is not a lock
across both the default and user files. Limits are 64MiB per file, one million
entries, 256 columns per record, 128KiB UTF-16 field units and 4096 filename
codepoints. Explicit callbacks allow worker cancellation during reading/parsing
and similarity computation. Tests also exercise fresh lookup after edits/deletion,
invalid UTF-8, cancellation, changed-source rejection and legacy file hashing.
