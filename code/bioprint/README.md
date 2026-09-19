# BioPrint — login that checks *how* you type, not just *what*

A website login that authenticates people by behaviour. Someone with the
correct password but the wrong typing rhythm is **blocked**, with no OTP or
second step. Scripts, headless browsers and replayed recordings are caught as a
separate fraud signal. Every decision is explained in plain English and lands
in well under 200 ms.

Built for ROOT 36 · Event 2 (BioPrint: Behavior-Based Login Security).

## Run it

```bash
conda env create -f ../typing/environment.yml     # first time only: Python 3.12, fastapi, numpy, playwright
conda activate bigidea
playwright install chromium                       # only needed for the bot demo

cd code/bioprint
uvicorn server:app --reload                       # http://localhost:8000
```

| Page | What it is |
|---|---|
| `http://localhost:8000/index.html#enroll` | create an account and teach BioPrint your rhythm (1 practice + 10 counted reps, **click** the button) |
| `http://localhost:8000/index.html#login` | sign in; allow / block with reasons, per-signal gauges and latency |
| `http://localhost:8000/dashboard.html?user=NAME` | live view for judges: every attempt, four signals kept separate, top deviating features, labelling |

Tests (128, ~15 s): `python -m pytest -q`

## The demo

1. **Genuine**: the owner enrols, then signs in → allowed, ~80 ms.
2. **Impostor**: a teammate types the *correct* password on the same laptop → blocked,
   e.g. "typing rhythm is unlike the account owner (5.8 vs limit 1.9) — held 'L' 88 ms longer than usual".
3. **Bots** (the server must be running; use the owner's real password):
   ```bash
   python attacks/scripted.py --url http://localhost:8000 --user NAME --password PW --mode fixed      # naive script
   python attacks/scripted.py --url http://localhost:8000 --user NAME --password PW --mode humanlike  # careful script
   python attacks/replay.py   --url http://localhost:8000 --user NAME --password PW --db bioprint.db  # replays a stolen recording
   ```
   The naive script trips the bot rules (webdriver, 5 ms key holds, identical intervals).
   The careful script passes the bot rules and is caught by the rhythm instead — two independent nets.
   The replay matches a stored sample to 0.00 ms; humans never repeat themselves within 4 ms.
4. A headless Chromium driving the real page is blocked the same way (`navigator.webdriver`, HeadlessChrome, timing variance 1%).

## How it works

```
browser                                server (FastAPI + SQLite)
  capture.js   key events ─────┐         features.py   pair keydown/keyup → H, DD, UD per key
  pointer.js   pointer path ───┼─POST──▶ pointer.py    14 reach-and-click features
  probe.js     30 env facts ───┘         bot.py        rules: automation, impossible timing, replay
                                         device.py     enrolled vs current environment
                                         scorer.py     scaled Manhattan vs the user's profile
                                         decide.py     signals side by side → allow / block
```

**Keystroke vector** (the primary signal), same layout as the CMU benchmark so
one scorer serves both: for a password of *k* physical keys, *k* hold times (H),
*k−1* down-down gaps (DD) and *k−1* up-down gaps (UD) → **3k−2** numbers.
Modifiers are excluded; a Backspace breaks positional alignment, so that sample
is asked to be retyped rather than scored.

**Profile**: per feature, the median of the 10 enrollment reps and the mean
absolute deviation about it (floored so a coincidentally constant feature cannot
dominate). **Score**: mean over features of `|observed − median| / spread`,
each capped at 6. **Threshold**: per user, 1.5 × the 75th percentile of
leave-one-out enrollment distances, clipped to [1.0, 1.9]. Block when score >
threshold. Each feature's share of the score becomes a reason.

Why scaled Manhattan and not a neural net: Killourhy & Maxion (2009) compared 14
detectors on the CMU keystroke dataset and it won. We reproduce their 9.6% EER
to the decimal with the naive version and reach 6.9% with the median/cap
variant. It fits in 1.6 ms, scores in 0.04 ms, and explains itself.

**Signals are never merged into one number.** A bot with the owner's exact
rhythm is a replay; a human with a different rhythm on the owner's laptop is an
impostor; the same rhythm from a new device is a new laptop. The disagreement
pattern is the information, so the dashboard shows four gauges.

## Reliability

Measured on the public CMU dataset (51 people × 400 reps of `.tie5Roanl`,
8 sessions on different days), `python -m eval.cmu`:

| Protocol | EER | Note |
|---|---|---|
| Published setup: 200 training reps | **6.9 %** | Killourhy & Maxion 2009 report 9.6 % for the same detector |
| Product setup: 10 enrollment reps, tested on *later sessions* | 20 % | 10.7 % when the enrollee already types the string fluently |
| Enrollment sweep 5 / 10 / 20 / 50 reps | 24.6 / 22.0 / 19.0 / 15.4 % | more reps buy a lot |

We report FRR at fixed FAR in the results JSON as well; EER alone hides which
side an error lands on. Live attempts are labelled genuine / impostor / bot on
the dashboard so the same numbers can be produced on real people.

## Privacy (read this)

Only `event.code` (which physical key) is stored, never the character, and never
the contents of any other field. **Hackathon-only trade-off:** on the password
field the key codes spell the password, so `bioprint.db` effectively stores
passwords and every wrong guess by keycode. This was a deliberate choice for a
36-hour demo and must not be reused in real work; the research design behind
this project stores per-user templates, not raw streams.

## Layout

```
server.py       API: register, enroll, login, attempts, labels, users
contracts.py    every shape the browser, engine and dashboard agree on
db.py           SQLite; raw samples are the record of truth, vectors are derived
engine/         features, scorer, pointer, bot, device, decide
static/         the pages and the three capture scripts
attacks/        scripted and replay attackers for the demo
eval/           CMU benchmark harness and results
tests/          128 tests
```

Research notes behind the design: `../../research/big_idea/` (Obsidian vault;
start at `14 BioPrint Hackathon.md`).
