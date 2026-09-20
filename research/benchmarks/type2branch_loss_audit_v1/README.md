# Set2Set generated numerical audit

Original Type2Branch loss class/function and official Addons v0.23.0
pairwise_distance are loaded as reviewed AST definitions. Function bodies are
unchanged; unused imports and Addons binary initialization are not executed.
The official source and Apache license are retained. This is not verification
of the entire Addons package or the original author's runtime.

The independent NumPy oracle averages hinge distances over identity pairs i<j,
within-identity pairs a<b, and every negative c. Distances are unsquared. The
radius penalty is beta times mean absolute deviation of each class radius
relative to the mean radius. Labels do not define membership: contiguous blocks
do. Identity/sample permutations can change the value; fixtures detect this.

Three hand fixtures, a random loss/finite-difference gradient fixture and a
published-cardinality K10/N15 fixture pass. Full report contains numeric errors.
Collapsed embeddings produce nonfinite loss and gradients in publisher source;
this result is retained, not repaired silently. A future training runner must
reject nonfinite updates and document any stabilization as an adaptation.

No dataset observations, fitting, optimizer update or recognition metrics were
part of this loss-only audit. A separate optimizer smoke artifact may provide
integration evidence, but does not change these scope limits.
