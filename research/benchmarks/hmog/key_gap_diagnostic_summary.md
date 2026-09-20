# First-fit key-gap diagnostic

A separately frozen, schema-only reread covered identity 717868, session 3 only. No feature arrays, models or held-out measurements were accessed. Fourteen full Activity-anchored windows were examined; retained spans use at most the first 52 strict uniquely matched pairs.

Across those spans, 252 extra recorded endpoints belonged to orphan/ambiguous cycles and ten belonged to complete cycles omitted by strict touch matching. There were no duplicate retained endpoints, unknown-action extras, or boundary-overlap extra endpoints. Retained endpoints represented 71.23–88.37% of raw span events, median 82.23%. These are structural categories; they do not establish why events are missing or whether an unpaired event corresponds to an intended character.

The four windows containing the required 52 retained pairs still had 19–25 orphan/ambiguous extra events each. Their retained/raw event ratios were 80.62–84.55%. Therefore treating consecutive retained pairs as consecutive physical keystrokes would conceal recorded gaps. A later protocol may explicitly model **intervals between successive retained observed pairs**, with gap metrics and no repaired events, but that is a preprocessing adaptation requiring a new freeze.

Candidate pure per-pointer parser `scripts/hmog_contact_parser.py` has six generated-data tests. It preserves real DOWN/POINTER_DOWN and UP/POINTER_UP endpoints; pointer count above one alone does not reject a contact. CANCEL invalidates every active pointer in the activity. Nonportrait events invalidate the affected contact, and duplicate events or repeated downs invalidate a cycle. Missing endpoints are never synthesized. This parser has not been used for participant feature extraction; no v3 extraction is authorized by this diagnostic.

Artifacts: `key_gap_diagnostic_plan.json`, `key_gap_diagnostic_source.py`, and `key_gap_diagnostic_results.json`. Window indices restart within each activity; report order follows the original activity file. Strict v2 artifacts remain unchanged.
