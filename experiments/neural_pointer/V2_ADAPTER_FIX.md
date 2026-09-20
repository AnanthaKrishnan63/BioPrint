# V2 stationary-coordinate correction

V1 reproduced the old policy exactly (FA14/90, FR13/60) but the new endpoint
rejected19 of33 pointer captures because it required600 nonzero displacements.
That was an invented input guard, inconsistent with the trained model's native
641-coordinate contract, which includes stationary/button events.

Remove that guard; still reject entirely stationary streams, missing/short data,
untrusted/touch events, invalid timestamps and duplicate recordings. Keep all
source points, encoder weights, profiles, threshold, account mappings and cases
unchanged. Rerun the same paired API comparison and retain V1. This corrects an
adapter defect after exposure; it does not create an untouched evaluation or
prove that mouse captures are unforgeable. V2 results are exploratory.
