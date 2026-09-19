---
title: Account Holder Verification — Datasets and Data Availability
tags:
  - big-idea
  - identity
  - research/data
updated: 2026-09-18
---

# Datasets and data availability

## Summary

**Keystroke data is abundant and open; everything else is scarce.** Fixed-text and free-text keystroke datasets are mature, well-benchmarked and downloadable, including one with 136 million keystrokes from 168,000 subjects. Mouse-dynamics datasets are far smaller and mostly request-gated. Browser-fingerprint datasets essentially **do not exist in public form** — the data is commercially valuable and privacy-sensitive, so every fingerprinting paper collects its own. Most importantly, **no public dataset holds the device axis constant while varying the person**, which is precisely the design this project needs.

> [!important] The gap that defines the project
> There is no public dataset where *multiple people use the same physical machine and browser profile*, and no public dataset pairing **device fingerprint + behaviour + context** for the same sessions. Collecting one is both the main practical cost of this project and its clearest research contribution. See [[big_idea/09 Experiment Plan|Experiment 3]].

## Keystroke dynamics — fixed text

| Dataset | Finding summary | Access | Limitation / best use | Link |
|---|---|---|---|---|
| **CMU Keystroke Dynamics** (Killourhy & Maxion, DSN 2009) | 51 subjects typing the password `.tie5Roanl` **400 times across 8 sessions on separate days**. The session structure is what makes it uniquely valuable — it permits honest cross-session evaluation. | **Open:** direct download. | Single fixed string, single keyboard, lab conditions, 2009-era timing resolution. **Still the correct first benchmark.** | [CMU keystroke](https://www.cs.cmu.edu/~keystroke/) |
| **KeyRecs** (2023) | Modern keystroke dynamics and typing-pattern dataset published as a data paper with documented collection protocol. | **Open.** | Newer collection conditions; smaller community baseline than CMU. | [Data in Brief](https://www.sciencedirect.com/science/article/pii/S2352340923006091) |
| **Synthesised free-text keystroke features** (2023) | Paired human-written and synthesised samples — useful for testing whether a detector can spot **generated/replayed** timing. | **Open.** | Directly relevant to the replay-attack threat model. | [PMC10139888](https://pmc.ncbi.nlm.nih.gov/articles/PMC10139888/) |

## Keystroke dynamics — free text

| Dataset | Finding summary | Access | Limitation / best use | Link |
|---|---|---|---|---|
| **Aalto 136M** | **136 million keystrokes from 168,000 subjects** over three months, via an online typing test — fifteen transcribed English sentences, typed as fast and accurately as possible. Enormous *breadth*. | **Open.** | Shallow per user (one sitting), transcription task only, web-collected so conditions vary. **Ideal for pre-training an embedding model**, then fine-tuning per user. | [Aalto 136M](http://userinterfaces.aalto.fi/136Mkeystrokes) |
| **Clarkson II** | 12.9 million keystrokes across 103 users, averaging ~125,000 keystrokes each — **collected in the wild over months**, and including **mouse events and active program context** alongside keystrokes. Extraordinary *depth*. | **Request:** available from the authors on request. | The closest public analogue to real continuous-authentication conditions, and the only one combining keystroke + mouse + context. **Worth the request email.** | [Clarkson CITeR](https://citer.clarkson.edu/) |
| **Buffalo free-text** | 148 participants, 2.14 million keystrokes, both transcription and free-composition tasks, multiple sessions. | **Open / registration.** | Well-used benchmark; good middle ground between Aalto's breadth and Clarkson's depth. | [CUBS Buffalo](http://cubs.buffalo.edu/research/datasets) |

## Mouse dynamics

| Dataset | Finding summary | Access | Limitation / best use | Link |
|---|---|---|---|---|
| **Balabit Mouse Dynamics Challenge** | The de facto standard mouse-dynamics benchmark; released for a public challenge. Small subject count. | **Open:** GitHub. | Only ~10 users, remote-desktop capture conditions. Reported EERs on it are **not** representative of web conditions. | [GitHub](https://github.com/balabit/Mouse-Dynamics-Challenge) |
| **DFL / Chao Shen datasets** | Mouse-behaviour datasets used widely in the Chinese-language literature. | **Request / varies.** | Check collection conditions carefully before comparing to web numbers. | — |
| **TWOS** (The Wolf of SUTD) | Insider-threat dataset with mouse and keyboard traces from a gamified multi-user scenario. | **Request.** | Designed for masquerade/insider detection — conceptually the closest to the "stranger at the same machine" threat. | [SUTD](https://www.sutd.edu.sg/) |
| **Clarkson II** | (see above) includes mouse events. | **Request.** | Best available combined keystroke+mouse source. | — |

## Mobile / touch / motion

| Dataset | Finding summary | Access | Limitation / best use |
|---|---|---|---|
| **HMOG** | Hand movement, orientation and grasp — accelerometer/gyroscope with touch during reading and typing on phones. | **Open.** | Collected natively, not in-browser; browser `devicemotion` sampling rates are lower and permission-gated on iOS. Use for feasibility, not for transfer. |
| **BrainRun, Touchalytics** | Touch-gesture datasets (swipes, strokes) for continuous mobile authentication. | **Open.** | Same native-vs-browser caveat. |

## Browser fingerprinting

| Source | Finding summary | Access | Limitation / best use | Link |
|---|---|---|---|---|
| **AmIUnique** | Public fingerprinting demonstrator and research project; publishes statistics on attribute distributions. | **Statistics open; raw dataset request-gated.** | Use published distributions to estimate entropy priors without collecting anything. | [amiunique.org](https://amiunique.org/) |
| **EFF Cover Your Tracks** (formerly Panopticlick) | Live uniqueness tester reporting bits of identifying information per attribute. | **Open tool; aggregate stats only.** | The standard reference for per-attribute entropy figures. | [Cover Your Tracks](https://coveryourtracks.eff.org/) |
| **FingerprintJS OSS** | Not a dataset — a **collector**. Produces the attribute vector you then gather yourself. | **Open source.** | The practical route to building your own dataset. | [GitHub](https://github.com/fingerprintjs/fingerprintjs) |
| **CreepJS** | Research demonstrator that also reports which signals are internally inconsistent (spoofed). | **Open source.** | Reference for consistency-check features. | [CreepJS](https://abrahamjuliot.github.io/creepjs/) |

> [!note] No public browser-fingerprint corpus
> Every fingerprinting study collects its own data. Plan on the same: a static collector page, a subject pool, and repeat visits over weeks. This is genuinely low-cost — the expensive part is **repeat visits**, which is what measures drift.

## Network and context

| Source | Finding summary | Access |
|---|---|---|
| **MaxMind GeoLite2** | Free IP → country/city/ASN database. Sufficient for impossible-travel and ASN-familiarity features. | **Open, registration.** |
| **IP2Location LITE**, **IPinfo free tier** | Alternatives, including proxy/VPN/hosting classification on paid tiers. | **Open / freemium.** |
| **JA4+ fingerprint database** (FoxIO) | Reference TLS/HTTP client fingerprints for known clients. | **Open source.** |

## Collection plan for this project

Since the critical data does not exist publicly, the schema to collect:

```jsonc
{
  "session_id": "uuid", "subject_id": "hash", "device_label": "A|B|C",
  "ts": 1758153600000, "condition": "genuine|cross_device|cross_person|both",
  "device":      { /* ~40 fingerprint attributes, raw values — never pre-hashed */ },
  "server_side": { "ja4": "...", "h2_fp": "...", "header_order": [],
                   "asn": 0, "ip_class": "residential|mobile|datacenter|vpn",
                   "geo": { "cc": "", "city": "", "lat": 0, "lon": 0 } },
  "keystrokes":  [ { "code": "KeyT", "type": "down", "t": 12345.6 } ],  // codes only, never characters
  "pointer":     [ { "x": 0, "y": 0, "t": 0, "buttons": 0, "ptype": "mouse" } ],
  "motion":      [ { "ax": 0, "ay": 0, "az": 0, "t": 0 } ],
  "challenges":  [ { "type": "scrambled_keypad", "probe_id": "", "events": [] } ],
  "interaction": { "paste_fields": [], "autofilled": [], "nav_path": [], "blur_count": 0 }
}
```

**Collection rules**
- Store **raw event streams separately from derived features** — you will want to re-extract features after the first modelling pass, and re-collection is expensive.
- Log `event.code`, **never** `event.key` or the typed character. This is both a privacy requirement and removes any keylogging exposure.
- Downsample pointer events to ~60–125 Hz before storage; a two-minute session is otherwise ~15,000 events.
- Record the **browser build and OS version** with every session — feature distributions are not portable across browsers because of differing timer coarsening.
- Tag every session with its **2×2 condition** at collection time. Reconstructing it afterwards is unreliable.

## Related notes

- [[big_idea/02 Papers|Papers]]
- [[big_idea/07 Browser Signal Catalogue|Browser signal catalogue]]
- [[big_idea/09 Experiment Plan|Experiment plan]]
