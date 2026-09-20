# Source-translated synthesis input cleanup

Pinned Program.LoadDataset performs ThresholdPartitioner then CleanFTs on
training inputs, as well as cleanup following synthesis. The earlier output-only
interpretation was incomplete. Population models see the cleaned values.

Translated cleanup first records FT>1500ms partition indices, maps negative
HT/FT to1500ms, caps remaining HT/FT to1500ms, and sets first/partition FT to
int.MinValue. This is source translation tested on generated fixtures, not C#
runtime parity. It must not be substituted for neural base-channel clipping.

Applied only to the frozen705fitting windows/47identities:953negative inputFT,
213HT values above1500ms,678FT values above1500ms,678pause positions,1379invalid
outputFT positions (first-event/pause overlaps counted once). Original signed
and neural base arrays remain unchanged. OutputSHA is recorded in report.json.

Further source contracts: context selection requires at least10observations;
longest qualifying context wins. Context starts/resets with0xFF and has a
maximum7preceding-key bytes under the configuration. MemoryStorage.GetBulk
skips hash0. Mean synthesis truncates tointeger and uses random fallback for
missing context. Exact runtime/RNG behavior and paper-specific synthesis mode
remain unverified. These arrays are not synthetic outputs or residual features.
