# Fitting-only finite-context population model

All705frozen sequences from47fit identities feed separate HT/FT models.
Input cleanup and pause masks were frozen previously. This implements reviewed
mean-synthesis source semantics: initial/reset FF sentinel, context0–7,
feature/order/hash-specific storage, int.MinValue exclusion, sequential moments,
longest context with at least10observations, hash0 lookup exclusion, and integer
mean truncation. Nine generated tests cover boundaries, backoff and arithmetic.
The validated input domain0–1500ms avoids C# int32 squaring overflow. No negative
variance stabilization is added. This is source translation, not .NET parity.

Fit completed in2.124seconds with164,156models. The unpartitioned-key-query
coverage follows the reference synthesis dummy, rather than using observed
query timings for context resets. Each channel has61unmatched positions out
of70,500fitting positions. No fallback value was invented; missing predictions
remain NaN internally and order-1 in the saved coverage artifact. Serialized
population and coverage SHA256s are in report.json.

Average synthesis is a source-backed candidate, not a verified paper mode;
the CLI default is Histogram. Random fallback, final synthesis cleanup and
residual-channel conversion still need validation. This is a population fit,
not encoder training or recognition evaluation. All selection/calibration/DEV/
test observation values remain unaccessed by this stage. Prior exposure of
these fitting observations is documented in earlier audits.
