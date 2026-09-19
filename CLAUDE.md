# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Account holder verification

Detecting whether the **account holder** — or a different person, or a different
computer — is using a session, using only data reachable from a **web browser**.

```
hackathon/
├── code/
│   └── typing/  the data-collection harness (see its README.md)
│       └── verify/  kernel-vs-browser timing check (its own README.md)
└── research/    Obsidian vault. Open THIS directory as the vault root,
                 or the [[big_idea/...]] wikilinks will not resolve.
```

**Start here:** `research/big_idea/13 Roadmap.md` (**the plan** — where we are,
where this ends, and the decision gates), `research/big_idea/00 Home.md` (the
argument), `research/big_idea/10 Basics and Glossary.md` (acronyms, from scratch).

## Current focus: BioPrint hackathon

A 36-hour build of a website login that **blocks** a user who has the correct password
but the wrong behaviour, and flags bots (ROOT 36 Event 2, brief in
`ROOT 36 Problem Statements.pdf`). **Deadline ≈ 2026-09-21 00:00 local.** Plan, phases
and file ownership: `research/big_idea/14 BioPrint Hackathon.md`.

The app is `code/bioprint/`: `server.py` (API), `contracts.py` (pydantic models),
`db.py` (`bioprint.db`), `engine/` (features, scorer, bot, pointer, decide),
`static/` (pages + capture), `eval/` (CMU dataset EER), `tests/`.

```bash
cd code/bioprint && uvicorn server:app --reload
python -m pytest code/bioprint/tests
```

**Hackathon overrides** (hackathon only; never carry them back into the research):
- The product **blocks**. The brief forbids OTP or any friction fallback.
- `event.code` **is** logged on the password field, so `bioprint.db` effectively
  stores passwords by keycode. This was a deliberate choice; never reuse it in real work.
- Bot/replay detection is a required signal, **kept separate** from the behaviour score.
- The device-fingerprint axis is reduced to an automation probe feeding bot detection.

## The idea in one table

"Different person" and "different computer" are two detections, and no single
signal family covers both:

|  | Same device | Different device |
|---|---|---|
| **Same person** | normal login | new laptop → device signals fire, behaviour agrees. **Must not block.** |
| **Different person** | unlocked laptop, RAT, cookie theft. Fingerprint is perfect. **Only behaviour fires.** | both fire. Easy case. |

So: two orthogonal axes (device/context + behaviour), fused, with passkeys
short-circuiting. The **disagreement pattern between axes** is the most
informative feature, so never collapse the score to one scalar.

## Running it

```bash
conda activate bigidea          # ~/miniconda3/envs/bigidea — never the system python
cd code/typing && uvicorn server:app --reload   # http://localhost:8000
node --test                                  # metrics.js unit tests (19, all passing)
node --test --test-name-pattern rollover     # a single test, by name substring
curl localhost:8000/api/sessions             # recent sessions + metrics, for eyeballing

# kernel cross-check (from code/typing/verify)
sudo ~/miniconda3/envs/bigidea/bin/python evdev_logger.py -o kernel.jsonl
python compare.py --kernel kernel.jsonl [--session N]   # newest session if omitted
python make_synthetic.py --jitter 2 --drift 0.5 --drop 3 \
  && python compare.py --kernel synthetic-kernel.jsonl --browser synthetic-browser.json
```

Env is Python 3.12 + fastapi + uvicorn + evdev. Recreate with
`conda env create -f code/typing/environment.yml`. No linter or build step.

## How the pieces fit

`capture.js` → `metrics.js` → `app.js` → `POST /api/session` → `server.py` → `sessions.db`

- `capture.js` only records `{code, type, t}`; `t` is ms since the session's
  **first key event**, not page load. It drops `ev.repeat` and flags paste.
- `metrics.js` is pure (no DOM), so everything in it is testable under plain
  `node --test`. New features belong here, not in `app.js`/`index.html`, which are
  throwaway UI.
- `server.py` stores raw events in `keystrokes` and only the metric names listed
  in `METRIC_COLUMNS` into `session_metrics`; unknown names are silently dropped.
  Adding a metric means adding it to `METRIC_COLUMNS` **and** the schema — and
  `CREATE TABLE IF NOT EXISTS` does not migrate the existing `sessions.db`, so an
  `ALTER TABLE` is needed (never recreate the DB; it holds real recordings).
- `verify/compare.py` reads the browser side from `sessions.db` (or `--browser`
  JSON) and aligns it to `kernel.jsonl` by time.

## Conventions that are not negotiable

- **Log `event.code`, never `event.key`.** Physical key, never the character, and
  never the textarea contents. Privacy, and it keeps this from being a keylogger.
- **Raw events are the record of truth.** `keystrokes` holds them; `session_metrics`
  is a convenience copy of what the page displayed. Feature extraction *will*
  change — recompute from raw rather than re-collecting from people.
- **Never hash a fingerprint into one ID.** Browsers auto-update monthly and change
  canvas/UA-CH/GPU strings. Fuzzy-match attribute vectors instead.
- **Behavioural templates are per (user, device class).** A desktop template cannot
  score a phone session. This is a correctness requirement, not a refinement.
- **Report FRR at a fixed low FAR, never EER**, and respond with *friction*, not
  denial. At realistic base rates (1 impostor in 10⁴–10⁶) a 1% error rate means
  ~100 false alarms per true catch.
- Behavioural data identifying a person is **GDPR Art. 9 special-category** and
  covered by India's DPDP Act. Store templates, not raw streams, in anything real.

## State of the work

**Built and verified.** Browser capture (`capture.js`), 13 session metrics
(`metrics.js`, 19 tests), FastAPI + SQLite collector, and a kernel cross-check
that proves the browser's clock is trustworthy.

**Measured on this machine** (Firefox 155 / Ubuntu / X11) — full results in
`research/big_idea/12 Measurements.md`:
- dwell accurate to **2.8 ms at p95** = 1.4% of a typical dwell. Timer coarsening
  is real (exactly 1.0 ms) but not a threat.
- subject's dwell is **189 ms**, roughly double the literature, and it is *real* —
  explained by holding 2+ keys at once **32%** of the time.
- Firefox emits **phantom keydown events** in ~1% of the stream. Undocumented
  anywhere in the surveyed literature. Pairing logic must tolerate a duplicate
  keydown.

**Not built yet.** Per-key and per-digraph features, fixed-passphrase mode, any
pointer capture, the fingerprint collector, scoring/fusion, the active challenges.

> The 13 current metrics are **session-level averages, which cannot identify
> anyone**. Mean dwell is one number thousands of people share. There are
> currently **zero identity features**. The plumbing is done; the features are not.

## Next steps, in order

**During the hackathon, follow `research/big_idea/14 BioPrint Hackathon.md` instead.**
The list below is the long-term research order.

1. **Per-key dwell + per-digraph flight** over the 5 existing sessions. No new
   recording needed.
2. **Fixed-passphrase mode** (`.tie5Roanl`, CMU protocol).
3. **One more person types it** → the first EER. **This is the critical path** —
   everything downstream is guesswork until this number exists.
4. In parallel: the fingerprint collector, because it is gated on four weeks of
   repeat visits rather than on effort.

**Full plan with dependencies and decision gates:
`research/big_idea/13 Roadmap.md`.** Target architecture:
`research/big_idea/11 System Design.md`. Experiment designs:
`research/big_idea/09 Experiment Plan.md`.

Three gates are written down in advance, so the calls are not made
retrospectively. **Gate A** (after step 3): cross-session EER above ~20% means the
password vector cannot carry identity alone and weight shifts to challenges and
device. **Gate B**: a 5-second challenge must beat ~60 s of passive observation,
or the visible challenges get dropped. **Gate C**: behavioural EER *conditioned on
an identical device fingerprint* — the whole thesis. A null result there is still
a real contribution.

## Gotchas learned the hard way

- `sudo $(which python) …` **fails** — Ubuntu has no bare `python` outside conda,
  so it expands to nothing. Use
  `sudo ~/miniconda3/envs/bigidea/bin/python evdev_logger.py -o kernel.jsonl`.
- **Recordings are expensive; never overwrite one.** A synthetic generator once
  clobbered a real kernel log. Synthetic output now uses `synthetic-*` names and
  the logger refuses to write over a non-empty file. Copy before you experiment.
- **Do not align event streams by sequence shape** (`difflib`). Typing repeats the
  same keys constantly, so one dropped event makes a shape matcher lock onto the
  wrong repetition and invent dozens of losses. Anchor on **time**.
- **`pkill -f "uvicorn … --port 8000"` kills your own shell**, because the shell's
  command line contains that string. Find the PID from `ss -ltnp` and kill that.
- **Bind the demo server to `127.0.0.1`** unless a specific device needs it. On
  campus wifi `0.0.0.0` exposes register/enroll/login and the dashboard to
  thousands of clients with no rate limit. Set `BIOPRINT_SECRET` so sessions
  survive restarts.
- **Never `git commit` chained after `pytest … | tail`**: the pipe's exit status
  is `tail`'s, so a failing suite still commits. Use `set -o pipefail` or check
  the summary line first.
- `code/typing/README.md` points research notes at `~/.yoyo/arshad_mfsdsai/big_idea/`;
  that path is stale — the notes are in `research/big_idea/` here.
- `code/typing/verify/__pycache__` at the source was root-owned from a sudo run
  and is excluded here.
- A comparison tool that cannot detect *injected* error proves nothing about real
  error. `make_synthetic.py` exists for exactly that check — run it after touching
  `compare.py`.

## Writing research

Research goes in `research/` as Obsidian notes matching the existing conventions:
YAML frontmatter (`title`, `tags`, `updated`), `## Summary`, dense tables with a
finding-summary column and a link column, `> [!important]` / `> [!warning]`
callouts, and a `## Related notes` section of wikilinks. Mark unverified facts
(patent assignees, vendor claims) as unverified rather than asserting them.
