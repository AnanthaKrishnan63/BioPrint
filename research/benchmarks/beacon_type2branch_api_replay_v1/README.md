# Paired DEV fusion API replay

Actual isolated ASGI replay completed for all eight models and243 claims each:1,944 scores, maximum error0, zero decision flips across all three frozen thresholds. Pooled and per-probe identity metrics match exactly. TRAIN/test requests, external Host/Origin and nonloopback clients return403; malformed features422 and unknown models404.

No listening server. Encoders ran offline in the recorded preparation/inference stages; these routes replay the actual frozen fusion models on their paired features. This is not online browser capture or encoder inference. Negative DEV recognition results remain unchanged.
