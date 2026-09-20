# Additional frozen transfer check

Declared while the main TRAIN fitting run is in progress, before this check reads
any BEACON measurements. Use all three existing BEACON DEV identities, their
explicit TRAIN-enrollment recordings and DEV probes; leave test identities sealed.
Models/thresholds are the nine frozen CMU/KeyRecs artifacts, without any BEACON
population training, calibration, tuning or selection.

Use keyboard CSV only. Validate hashes and the existing record-role ledger.
Sort by elapsed press time, take disjoint ten-event windows, discard incomplete
tails, and calculate the identical 21 features in milliseconds. Enroll using the
first ten valid windows from the enrollment recording, evaluate every valid
probe window against every eligible identity. Reject nonfinite timings and
negative holds and count these exclusions. Never duplicate or pad windows.

Report FAR/FRR/EER and exact counts separately. Gameplay auto-repeat and arbitrary
key combinations differ from password input; only three identities are available.
These results are a domain-shift diagnostic, not login accuracy. Other datasets
contain pointer, touch, cognitive or mobile sensor data and cannot supply the
same desktop timing representation; do not mix them into typing training.
