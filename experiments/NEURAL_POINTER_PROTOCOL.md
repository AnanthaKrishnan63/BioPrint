# Trained-pointer integration and synthetic comparison

Freeze before running the new comparison. Use the unchanged login_combinations_v3
DEV cohort: fifteen synthetic profiles, sixty genuine and ninety impostor cases.
Compare the same typing-pointer code with versus without neural profiles installed.
Do not fit or change any threshold from these results.

Map synthetic accounts to fifteen distinct SapiMouse DEV identities using seed
20260920. These are artificial pairings with CMU identities, not linked people.
Each pointer profile is the frozen cosine center built from that SapiMouse
identity's separate three-minute TRAIN-support recording. Score native641-point
DEV windows through the actual exported encoder and new authentication endpoint.
Require encoder hash, source CRC, split/role and frozen score parity. No sealed
test payloads. No live database or listener.

For genuine cases, use the mapped owner's pointer window. For impostor cases,
use the next mapped identity's window. The existing keypad-match attack gives
the impostor owner-like target behavior, so it receives the owner's pointer too.
Keep original typing, bot, device and mobile-keypad inputs identical. Sample one
available window deterministically per case; never choose by its score.

Run each scenario from an identical enrollment-only database snapshot, so reused
public recordings represent independent counterfactual scenarios rather than
replay attacks. Challenge time is advanced only in the isolated simulation;
production clock checks remain enabled. Record any short/unusable source capture
as incomplete, not as a match. Include it in failure-to-admit FRR and report it
separately. Never synthesize missing pointer points or concatenate sessions.

All enrolled desktop step-ups now use FCN+cosine@5 at the frozen calibration
threshold0.9480821490287781. Direct typing acceptance and hard blocks stay unchanged;
mobile still uses the existing keypad. Dataset pointer threshold is experimental
for the browser task. Report exact FAR/FRR, initial challenge rate, neural use,
incomplete checks and per-scenario outcomes. Compare old stored results as an
additional audit, not as a substitute for the paired API run.

Calculate score-level pairing sensitivity across100 deterministic remappings
using the same recorded source score pool. These repeated mappings explore
unmeasured cross-modal pairing; they are not independent people or confidence
intervals. A branched policy has no declared scalar EER; do not invent one.
