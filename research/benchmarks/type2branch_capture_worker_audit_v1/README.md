# Capture inference worker preparation

The new capture worker preserves explicit 25/50/75/100 true lengths and applies the corresponding float64 calibrated threshold to each row of mean-Euclidean scores. Its release is separately versioned from the original fixed-capture release and requires a pinned manifest, exact artifact/source inventory, profile metadata consistency and matching checkpoint optimizer iterations.

The builder requires completed DEV evaluation and checks gallery, probe identity/length alignment, offline decisions, feature hashes and checkpoint provenance before packaging. It does not activate an API pin. The worker serves bounded stdin/stdout JSON without opening a network listener.

43 generated wire/release tests passed in 0.32 seconds. Padding is checked before float32 conversion, including tiny values that would underflow to zero. An actual invalid-pin worker invocation rejected before TensorFlow/model loading. Real release construction, model inference, parent-client/API integration and full DEV replay remain pending. No new DEV or test payloads were decoded.
