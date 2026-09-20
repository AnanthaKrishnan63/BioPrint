# HMOG tap reference: explicit numerical adaptation

The [original HMOG paper, sections IV-A and IV-B](https://arxiv.org/html/1501.01199#S4), specifies tap duration, inter-press velocity, and nine contact-size descriptors: mean, median, standard deviation, quartiles 1/2/3, initial size, minimum, maximum. Median and second quartile are redundant; both remain in the eleven-feature schema. The paper uses pixel/second velocity, enrollment feature means and user standard deviations, and mean tap vectors over authentication scans. It discusses a minimum of 80 enrollment observations and feature selection; its best SM tap comparator retained three features. This implementation preserves the eleven-feature family, not that selected comparator or the full published protocol.

`scripts/hmog_tap_reference.py` provides pure functions with no dataset access:

- `tap11(contact, previous_contact, conventions=...)`: complete contact timestamps in event milliseconds, x/y in pixels, raw size samples. Caller supplies DOWN first and UP last, same session/frame, and genuinely consecutive contacts. Missing prior contact raises `ValueError`; upstream must report that tap as unscorable. No zero-velocity imputation, event repair, or physical size/pressure conversion.
- `scan_mean(tap_vectors)`: unweighted arithmetic mean; caller enforces time-window and session membership before calling.
- `fit_profile(enrollment_taps, conventions=..., min_taps=...)`: tap-wise enrollment mean/std, explicit minimum enrollment, and active-feature mask/count.
- `scaled_manhattan(profile, authentication_vector)`: sum of absolute standardized differences, larger means more dissimilar. Ignored dimensions contribute nothing. No usable enrollment dimensions means no verdict.
- `profile_json` / `profile_from_json`: finite, schema-checked serialization with feature order and numerical conventions.

Required `Conventions` records quartile interpolation (`linear`), standard-deviation degrees of freedom (`0` or `1`), Euclidean press-coordinate displacement, minimum spread, and `ignore` policy for spread at or below that bound. These details are explicit implementation choices where the paper does not fully specify numerics. Tap sample statistics include supplied endpoint samples, use equal sample weights, and do not interpolate a temporal trajectory. First-tap missingness, duplicate-time rejection, summed distance, and scan boundaries likewise require a frozen caller contract. This is an arithmetic adaptation, not an exact reproduction or a current SOTA claim.

Six synthetic tests cover hand-calculated eleven features, units, redundant median/Q2, missing/overlapping/cross-session contacts, mean/std distance, ignored dimensions, minimum enrollment, serialization, malformed profiles, and numerical overflow. No participant measurements were read, no model trained, and no evaluation performed. Root must freeze its model/feature contract before applying this module to HMOG records.
