# DEV runner audit before observation reads

Implemented gated feature preparation and offline inference for the frozen79identity protocol. Independent review caught full-session parsing beyond prescribed windows; now only required prefixes are decoded, with a generated check that a cut cannot masquerade as the real session terminal. This scope clarification is frozen here before actualDEVreads.

The preparer verifies the original active cohort, disjoint/exhaustive exposure groups, frozen innerTRAIN feature dependencies and population provenance. Any failure prevents a complete-cohort status and leaves metrics empty. The evaluator requires complete verified features, matching checkpoint and threshold, then reports pooled/macro metrics plus within-cohort and full-account claims grouped by probe identity.

Seventeen generated checks passed; both executable prerequisite gates refused before data/output access. Actual preparation/inference, checkpoint behavior, serialization and recognition metrics remain unverified until prerequisites complete.
