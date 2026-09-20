# Type2Branch legacy random audit

Source-translated seeded System.Random passes 200 official Next vectors from Microsoft and the corresponding reciprocal-scaled double checks. Thirteen combined RNG/context tests pass. Source downloads and licenses are pinned by commit and SHA-256.

The synthesis candidate uses global seed 1234 to seed the first thread, preserves the stream across sequences, and fills all HT positions before FT positions. Found models consume no random draw. Profile initialization seeds the stream but does not itself draw from the local stream; explicit AVERAGE does not call the randomized synthesizer selector. Independent source review found no arithmetic mismatch.

This does not establish CLR runtime parity, paper-mode fidelity, residual conversion, or recognition accuracy. No dataset observations were read.
