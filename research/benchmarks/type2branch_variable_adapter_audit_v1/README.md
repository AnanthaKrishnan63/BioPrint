# Variable-length adapter contract audit

This candidate retains actual 1–100-event windows, including final partial windows. It tracks true length explicitly, validates continuity before windowing, and treats a capped prefix as distinct from confirmed session termination. Population synthesis and fallback draws operate only on real events; residuals are computed before all five channels are padded to 100. Padding has false residual validity and context order -2; missing model context remains -1.

22 generated tests passed in 0.28 seconds. Tests cover author base preprocessing at lengths 1, 25, 50, 75, 99, 100, 101, 199 and 225; prefix exclusion and terminal semantics; cross-window continuity; actual zero-valued events; fallback draw counts; full-window feature and RNG-state equality with the existing pipeline; unchanged inputs; and invalid limits.

No dataset observations were read. No FAR, FRR or EER is claimed. The original failed DEV protocol remains unchanged. A separate protocol and calibration are required before evaluating this adapter on DEV.

The author merger mishandles single-row synthetic CSV input by replacing it with zeros. Preserving the real one-row HT residual is an explicit adaptation. Lengths below 25 also fall outside the current mixed-prefix training lengths. Training on prefixes of already synthesized features does not exactly reproduce shortened raw-input synthesis because fallback draw ordering can change.
