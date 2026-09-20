# Fitting-only ordering and boundary audit

Same47fitting identities/session1; source and role hashes pinned before decode.
All47incomplete records are final rows, one per fitting identity; none occurs
at the start or in the middle. This is observed tail structure, not proof of
the collector's terminal-record convention. Current/next event reconstruction
must distinguish a final event from a complete two-key digraph explicitly.

Across132,381complete finite-timing rows,2,116have negative DD and nonnegative
UU. One has nonnegative DD and negative UU; none has both intervals negative.
No inferred second hold (DU-DD) is negative. Consequently assuming strict press
order would reject2,116rows, while assuming strict release order is also not
universally supported. Neither reordering nor clipping has been performed.

Two generated boundary/order tests pass. All prior row/complete/negative counts
reproduce. Source hashes verified. These observations do not establish task
boundaries or the original ordering algorithm; no model-ready sequences or
cohort change were emitted. Selection/calibration/DEV/test timings stayed opaque.

An independent primary-source search found no accessible collector code or
explicit ordering/tail specification. The original paper and Zenodo description
state that the custom website computes digraph timings but do not establish
these details in accessible material. Official full-text endpoints were not
accessible through the browser tools; no access restriction was bypassed.

Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC10474054/ and
https://zenodo.org/records/7886743 . The later Zhang-0124/client-side-keystroke-ca
repository is downstream analysis, not the original collector.
