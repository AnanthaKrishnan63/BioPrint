# Fitting-only invalid-record classification

The same47TRAIN-fitting identities/session1 were scanned under source and role
hash checks. Earlier totals reproduce exactly:132,428rows and57undecodable rows.
Among132,371previously decodable rows, DD is negative2,111times and zero3times.
The earlier nonpositive interval total was2,114; these are not all ties.

There are10CSV key parsing errors,47empty-key records and188missing timing
fields.132,381rows have five finite timing fields, including the malformed-key
cases; their DD counts are2,116negative and3zero. Categories overlap, so do not
sum diagnostic counters as though they were mutually exclusive row counts.

All661,905finite timing fields in complete rows lie within1e-6milliseconds of
an integer after seconds-to-milliseconds conversion. This measures numerical
wire compatibility; it does not establish recorder clock accuracy. No rounding
or data repair was performed. Three generated classification tests pass.

The next event-order policy still needs source evidence. Missing key/timing
positions, negative intervals and quote escaping must not be silently repaired
or concatenated. No eligibility/cohort changes, model fitting or selection,
calibration, DEV or test observation decoding occurred.
