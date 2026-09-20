# Runtime models

These assets are sufficient for the final app; no training dataset download is needed.
`launch.py` verifies their SHA-256 hashes against `manifest.json` and runs a CPU
inference check before serving.

| Asset | Purpose | Origin |
|---|---|---|
| `pointer-encoder.pt` | Frozen TorchScript fully convolutional encoder; 128-dimensional movement embeddings | BioPrint training on the SapiMouse representation-training split, epoch 27 |
| `typing-background.json` | Separate negative fitting and calibration bank for compatible typing profiles | 12 public CMU identities, disjoint from the synthetic evaluation accounts |

The pointer profile is fitted from each user's three one-minute enrollment runs.
The deployed cosine threshold is 0.9480821490287781. This is a trained encoder,
not a database of existing app users.

Typing uses a per-user radial basis function support vector machine when the
feature schema matches the public `.tie5Roanl` benchmark and sufficient enrollment
is available. Other passwords use the statistical enrollment-distance scorer.
The bank contains a small public-data-derived feature subset (not the full CMU
dataset). It is a necessary calibration asset, not universal password support.

## Attribution and provenance

CMU: Kevin Killourhy and Roy Maxion, *Comparing Anomaly-Detection Algorithms for
Keystroke Dynamics*, DSN 2009. [Dataset and citation](https://www.cs.cmu.edu/~keystroke/).

SapiMouse: [dataset authors' site](https://www.ms.sapientia.ro/~manyi/sapimouse/).
The training and calibration implementation is in
[`scripts/pointer_sapimouse_benchmark.py`](../scripts/pointer_sapimouse_benchmark.py)
and the fixed application adapter in
[`pointer_neural.py`](../code/bioprint/pointer_neural.py).
Upstream reference source and its Apache-2.0 notice are preserved in
[`research/benchmarks/pointer_author_source/`](../research/benchmarks/pointer_author_source/).
Source datasets and third-party reference implementations retain their own terms;
this repository does not relicense them. No live participant recordings or local
account database are included.
