---
title: Measurements — Own-Machine Results
tags:
  - big-idea
  - identity
  - research/measurement
updated: 2026-09-19
---

# Measurements

Results actually measured in this project, as distinct from figures quoted from the literature. Everything here was produced on one machine and should be treated as a single data point until replicated on other browsers and hardware.

## Summary

The browser's clock is **trustworthy for this research**: dwell times are accurate to within about 3 ms at the 95th percentile, against signals of 20–200 ms. The timer-coarsening concern quoted throughout [[big_idea/07 Browser Signal Catalogue|the signal catalogue]] is real — Firefox quantises to exactly 1 ms — but on these numbers it is not a threat. Two unexpected findings came out of the same exercise: the subject's dwell times are roughly **double** the figures in the literature, and the browser occasionally **emits phantom keydown events** that no physical key produced.

---

## Method

Two recordings of the same typing, compared.

```
physical key -> kernel driver -> X11 -> browser -> JS handler
                     ^                                ^
                evdev stamps here            event.timeStamp claims here
```

The kernel stamps each key inside the input driver, before X11 and before the browser, which makes it the reference. Tooling is in `typing/verify/` — `evdev_logger.py` records the kernel side, `compare.py` aligns the two streams and reports.

> [!note] The alignment is the hard part
> The two clocks count from different moments and the kernel log runs longer at both ends, so the offset is unknown and large (here, +9382 ms). Matching by sequence *shape* fails badly — typing repeats the same keys constantly, so one dropped event makes a shape matcher lock onto the wrong repetition and report dozens of phantom losses. The working approach anchors on **time**: every pair of same-key events votes for the offset it implies, genuine correspondences all vote alike, and coincidences scatter.
>
> `make_synthetic.py` injects known faults so the tool can be checked against a known answer. It recovers 2 ms of injected jitter as 3.2 ms of dwell spread (theory: 2·√2 = 2.83), finds exactly the number of dropped events injected, and recovers 0.5 ms/s of drift as 0.486. **A comparison tool that cannot detect injected error proves nothing about real error.**

**Setup:** Firefox 155.0, Ubuntu, X11, 1920×1200 @ 1.5 DPR, internal laptop keyboard ("AT Translated Set 2 keyboard"). Session of 361 browser events / 364 kernel events over ~45 s, 353 matched.

---

## 1. Is the browser's clock trustworthy? — Yes

| Measure | Result |
|---|---|
| **Dwell error** (browser − kernel) | mean **−0.65 ms**, sd 1.33, p95 **2.82 ms** — **1.4% of a typical dwell** (n=176) |
| **Flight error** | mean +0.66 ms, sd 1.41, p95 2.71 ms (n=175) |
| Per-event error | mean +2.04 ms, sd **1.07 ms**, p95 3.82 ms (n=353) |
| Drift | −0.004 ms/s — clock rates agree |
| Clock resolution | browser **1.0 ms** exactly; kernel ~1 µs |

The constant +9382 ms offset is simply the gap between starting the logger and the first keystroke, and it cancelled out of every interval exactly as predicted.

> [!important] Correction to the standard framing
> "A constant offset is harmless because it cancels" is only true if keydown and keyup are delayed **equally**. If releases were reported later than presses, every dwell would be inflated and nothing would cancel. So the **mean** of the dwell error matters as much as its spread. Measured at −0.65 ms, there is no such bias here — but it is the statistic to check, not the jitter alone.

---

## 2. Dwell times roughly double the literature — and they are real

Kernel ground truth, 181 keystrokes, modifiers excluded:

| | Measured | Literature (CMU and similar) |
|---|---|---|
| Mean dwell | **189.4 ms** | 70–110 ms |
| Median | 188.1 ms | |
| p10 / p90 | 139 / 241 ms | |
| Min / max | 111 / 312 ms | |
| Modifier dwell | 483.7 ms (n=4) | |
| Mean flight | 29.9 ms, **median −25.4 ms** | |
| Negative flights (rollover) | **108 / 180** | |

The working hypothesis before measuring was that Firefox reported keyup late, inflating dwell. **That is refuted** — the dwell error is −0.65 ms. The kernel agrees with the browser, so the long holds are genuine.

The mechanism is visible in the same data:

| Keys held simultaneously | Share of the session |
|---|---|
| 0 | 20.1% |
| 1 | 47.3% |
| **2** | **29.9%** |
| **3** | **2.7%** |

**A third of the time, two or more keys are held at once.** That one fact explains both anomalies together: dwell runs long because releases lag several presses behind, and 60% of flights are negative for exactly the same reason. This is a heavily "rolling" typing style.

> [!important] Why this matters for the project
> Both quantities sit far from the population norm, which is precisely what a discriminative feature looks like. **Overlap degree is not in the current metric set** and should be added — see [[big_idea/11 System Design|the free-text vector]]. It also means population-level priors from the literature should not be assumed when designing thresholds.

---

## 3. Phantom keydown events — ~1% of the stream

Four browser events in 361 had no kernel counterpart. They are not dropped events; they are **invented** ones. Kernel times shifted by the offset for comparison:

```
kernel                          browser
34: KeyO down  14.5836          34: KeyO down  14.5840
35: KeyN down  14.6946          35: KeyN down  14.6940
36: KeyO up    14.8493          36: KeyG down  14.8500   <- phantom, 5 ms early
37: KeyG down  14.8554          37: KeyO up    14.8540
38: KeyN up    14.8613          38: KeyG down  14.8550   <- the real one
                                39: KeyN up    14.8600
```

Firefox emitted `KeyG down` **twice with no release between**, about 5 ms before the genuine press. This is not OS auto-repeat — that fires hundreds of milliseconds apart and is filtered via `event.repeat`. All four occurrences were at points where keys overlap within the browser's 1 ms tick.

**Impact:** pairing ignores the duplicate but measures dwell from the *first* keydown, so that keystroke's dwell is inflated by the gap. At ~1% of events this threatens nothing, but it is worth knowing that a browser event stream is not perfectly faithful even when nothing is lost. No mention of this behaviour was found in the surveyed literature.

---

## Open questions

1. Does the phantom-keydown behaviour occur in Chromium, or is it Firefox-specific?
2. Does the ~2 ms mean per-event lag change under main-thread load? Every measurement here was on an idle page.
3. Is the 1.0 ms quantisation different under Firefox `resistFingerprinting`?
4. How much of the 189 ms dwell is the person and how much is this specific keyboard? Needs the same subject on a second device.

## Related notes

- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]] — the signals these results qualify
- [[big_idea/11 System Design|System design]]
- [[big_idea/09 Experiment Plan|Experiment plan]]
