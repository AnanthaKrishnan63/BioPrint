# Frozen production bot-rule validation

The plan and script hashes were frozen before dataset reads. `scripts/bot_rule_validation.py` evaluates the unchanged `engine.bot.check` against every strict CMU development row (sessions 5–6), using each account's first ten session-1 training recordings as prior vectors. It reconstructs events through the existing production replay adapter and derives features with production functions. No database or listener is involved; sessions 7–8 remain unparsed.

## Genuine development results

- 5,100 genuine recordings from 51 accounts.
- 15 false flags (0.294%), affecting 12 accounts (23.53% had at least one flag across 100 attempts).
- All 15 triggered `impossible_hold`. No genuine replay or other timing flags occurred.
- Empty browser environment produced the constant weak `no_probe` contribution (0.4) on every recording; this alone does not flag.
- Source-versus-reconstructed timing feature difference was at most 9.1e-13 ms.

`results.json` includes every rule's count, account-level clusters and individual decision records. These are bot-signal false flags, not complete-login FRR.

## Explicitly synthetic training stress checks

Each of 510 training enrollment recordings was cloned, with independent uniform event-timestamp jitter under a fixed seed. The stored priors remain original enrollment vectors.

| Jitter half-width | Flagged synthetic fraction |
|---|---:|
| 0 ms | 100% |
| 2 ms | 100% |
| 5 ms | 90.20% |
| 10 ms | 1.57% |

These are controlled input stress checks, not real bot recordings, unseen attacks, or heldout bot FAR. Jitter can alter event ordering; production pairing/rules handle the resulting synthetic stream without invented corrections.

## Limits

CMU contains timing data, not browser `isTrusted`, device probes, or pointer trajectories. Reconstructed events assume `trusted=True`; zero trust flags cannot validate that rule. Browser/environment and pointer rules remain unmeasured. Existing replay constants were historically developed using this legacy CMU corpus, so this is not independent or pristine-holdout specificity evidence. No rule or threshold was changed from these results, and no real-bot FAR/EER is claimed.

Run with `bigidea` and `PYTHONPATH=.research-deps:code/bioprint`. The script refuses to replace existing results; inspect the frozen plan and logs first.
