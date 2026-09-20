# Raw-prefix TRAIN calibration preparation

Completed preparation for sixteen reserved TRAIN calibration identities. Each of four artifacts contains 80 identical full-length gallery windows and 160 probes of exactly 25, 50, 75 or 100 real events. Raw events are truncated before population synthesis and residual construction; only then are all five feature channels padded. Gallery synthesis and each probe length use separate fresh first-thread random streams in sorted identity/window order.

Source, input and output hashes, identity counts, true lengths, zero tails, padding masks and fixed galleries were verified. The combined generated adapter/preparation/completion-gate suite passed 64 tests in 0.39 seconds.

No model inference or threshold fitting occurred. The frozen plan specifies one empirical 1% FAR threshold per exact probe length after the paired TRAIN checkpoint is selected. Calibration performance will not be DEV performance. No DEV or test observations were decoded. These are correlated same-session calibration variants, not independent trials; source preprocessing and random-stream adaptations remain explicit.
