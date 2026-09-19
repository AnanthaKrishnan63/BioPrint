---
title: "15 Scrambled Keypad"
tags: [bioprint, hackathon, design, keypad, step-up, cross-device]
updated: 2026-09-19
---

## Summary

A numeric captcha on a keypad whose digits are shuffled every time. The user is
shown a 6-digit target and taps it on a 0–4 keypad laid out in a fresh random
arrangement (3×2 grid, one blank cell, backspace in a fixed spot). Every tap is a
**visual search → reach → press**, so one captcha yields six behavioural samples
that exist on **any device, touch included**. That is the gap it fills: a touch
keyboard reports no key timing (see [[14 BioPrint Hackathon]]), so a phone had no
behavioural signal at all. It is enrolled at signup (6 captchas after the 10
password reps) and used as the **step-up on a new device** (2 captchas).

> [!important] The claim under test
> The *cognitive* part of the tap (how long a person takes to find and reach a
> digit on a scrambled layout, and their cadence) is hypothesised to transfer
> from mouse to thumb; the *motor* part (path, velocity shape, landing scatter)
> is known not to. So the vector is split in two, scored separately, and only the
> cognitive half may decide across device classes. This is a hypothesis to be
> measured on the project lead (laptop vs phone) plus one impostor; the CMU
> dataset cannot validate it.

## Design decisions (approved 2026-09-19)

| Decision | Choice | Why |
|---|---|---|
| Keys | 0–4 (`KEYPAD_DIGITS = 5`), one constant to flip to 0–9 | shorter, phone-friendly; 0–9 only if 0–4 cannot separate people |
| Target | 6 digits, no three-in-a-row | 5 transitions per captcha even with 5 keys |
| Captchas | 6 at signup, 2 at step-up | ~36 enrollment taps, ~12 step-up taps |
| Sample unit | the **tap**, not the captcha | 36 samples instead of 6 for the profile fit |
| Threshold | run-level leave-one-out (hold out a captcha, score its taps' mean) | a captcha score is a mean over ~12 taps, far tighter than one tap |
| Challenge | server-issued id + layout + target, 5-minute TTL, single use | a bot cannot choose its layout; a replay fails on the target |
| Corrections | wrong digit stays red until backspaced | corrections are real events, stored, not auto-fixed |
| Decisive? | cognitive half decides the step-up (allow/**block**); motor half advisory | the only behavioural signal a phone can produce |

## Two halves

| Half | Signal name | Features (per tap) | Scored when |
|---|---|---|---|
| Cognitive | `keypad` | interval from previous release (or layout shown) to this press; that interval per bit of Fitts difficulty; press duration | always |
| Motor, mouse | `keypad_motor` | reach kinematics as in `engine/pointer.py`: Fitts slope, path efficiency, curvature, velocity shape, overshoot, landing offset, hover/settle | device class = enrolled class |
| Motor, touch | `keypad_motor` | landing offset (fraction of key), hold | device class = enrolled class |

Both use the scaled-Manhattan scorer of [[14 BioPrint Hackathon]] unchanged.

## Routing (the step-up decision table, extended)

| New device detected, and… | Route |
|---|---|
| device class differs (phone ↔ laptop) | keypad |
| no key timing at all (touch keyboard) | keypad |
| rhythm passes but ≥ 80 % of its limit | keypad |
| after 3 more typings the median is still ≥ 80 % of the limit | keypad |
| otherwise (same class, comfortable pass) | type the password 3 more times |

Without a keypad profile nothing changes: typing step-up on a laptop, "retype" on a phone.

After the keypad: bot flag → block; cognitive score above its limit → block; else
allow, with motor and device shown as advisory.

## Bot rules specific to the keypad

Untrusted events; the environment rules of `engine/bot.py`; a tap interval under
120 ms on a freshly scrambled layout (a repeat of the same digit is exempt);
first tap under 200 ms; identical intervals; holds under 10 ms; taps at exact key
centres; a mouse-class run with no movement; a device class that contradicts the
browser's claim (touch taps with `maxTouchPoints 0`, mouse taps on a mobile UA).

## Laptop vs phone from the keypad itself

`pointerType` on the taps, the absence of any movement before a tap, landing
scatter and viewport size give a device class **independent of the fingerprint
probe**. The profile stores the enrolled class. A contradiction between the two
is a bot tell, not a device signal.

## Files

`contracts.py` (KeypadChallenge/Tap/Run/In, constants), `db.py`
(`keypad_challenges`, `keypad_runs`, `users.keypad_model`), `engine/keypad.py`
(features, profile, scoring, bot rules), `engine/decide.py` (routing, `decide_keypad`),
`server.py` (`/api/keypad/challenge`, `/api/enroll/keypad`, `/api/login/keypad`),
`static/keypad.js` (widget), `static/app.js` (enroll and step-up phases).

> [!warning] Limits
> Six enrollment captchas is thin. The cross-device claim is untested until the
> lead's phone runs are scored against the laptop profile. Five keys give only
> 120 layouts, so the layout is a weak replay defence on its own; the changing
> target is the real one.

## Related notes

- [[14 BioPrint Hackathon]] — the plan this extends
- [[11 System Design]] — two axes, fused, never one scalar
- [[10 Basics and Glossary]] — Fitts's law, scaled Manhattan, FAR/FRR
