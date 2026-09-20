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

## Share BioPrint on college Wi-Fi

Keep the normal localhost server on port 8000 for the host. From the repository
root, start a separate, restricted participant interface:

```bash
LAN_HOST=10.128.13.117 bash code/bioprint/run-lan.sh
```

Replace the address with your current Wi-Fi IPv4 address (`hostname -I`).
Participants open `https://<LAN_HOST>:8443/index.html#enroll` on the same network.
The browser first asks for invitation credentials: username `participant`, with
the random password stored in `code/bioprint/.lan/invite-password`. Share that
password privately with participants; it is separate from their BioPrint account.

HTTPS uses a self-signed, seven-day certificate. Verify its SHA-256 fingerprint
against the launcher's output through an in-person or trusted channel before
accepting the certificate on a participant device. Do not blindly bypass a
certificate warning. Campus Wi-Fi may isolate clients; if connection fails, use
a private hotspot rather than disabling the firewall or forwarding router ports.

The LAN interface requires the invitation for every request, binds only to the
chosen address, disables proxy-header trust, and limits requests and body size.
It blocks the dashboard, labeling, API documentation, and database downloads.
Signed-in users can retrieve only their own latest attempt. Enrollment status
omits behavioral statistics. Review all participants from `http://localhost:8000`.
Both instances use the existing `bioprint.db` by default.

Use **demo-only passwords**: the existing research database retains physical key
codes from password entry. Invitation holders can submit registrations and
authentication attempts; this is a controlled demo, not a hardened public service.
Stop sharing with Ctrl-C. To rotate the invitation, stop the server, delete only
`.lan/invite-password`, and restart. `.lan/` is private and ignored by Git.

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

The live scorer uses scaled Manhattan distance for small enrollment sets and
feature-level explanations. Separate research experiments compare supervised
and neural methods. Historical benchmark numbers used different protocols and
must not be presented as current held-out accuracy or a SOTA comparison.

**Signals are never merged into one number.** A bot with the owner's exact
rhythm is a replay; a human with a different rhythm on the owner's laptop is an
impostor; the same rhythm from a new device is a new laptop. The disagreement
pattern is the information, so the dashboard shows four gauges.

## Reliability

The strict CMU experiment uses sessions 1–4 for training, 5–6 for validation/dev,
and seals sessions 7–8. Legacy all-session loading is disabled. Earlier work had
already exposed later sessions; this cannot retroactively establish a pristine
test set. Current scripts never read sealed measurements.

With ten enrollment samples, frozen dev macro EER is **22.04%** for the live
distance baseline and **15.50%** for the training-selected RBF SVM. At thresholds
calibrated to 1% FAR on training data, actual dev FAR/FRR are 1.10%/74.20% and
1.00%/72.98%, respectively. The SVM is experimental and is not enabled for live
accounts. Low EER alone does not establish usable low-FAR authentication.

See [research protocol](../../research/benchmarks/PROTOCOL.md),
[experiment log](../../logs/EXPERIMENTS.md), and
[research commands](../../research/benchmarks/README.md) for datasets, limitations,
frozen artifacts, and API replay. All thresholds and model choices use training
data; dev validates frozen choices. No zero-error claim is supported.

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
