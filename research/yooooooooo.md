---
title: Big Idea — Original Brainstorm
aliases:
  - Account holder detection brainstorm
tags:
  - big-idea
  - research/brainstorm
updated: 2026-09-18
---

# Original brainstorm

Seed ideas for [[big_idea/00 Home|the big idea]]. Each is developed into a concrete, testable design in [[big_idea/08 Active Challenge Designs|Active challenge designs]].

1) Trick the user : test reaction and corrections patterns
2)  Patterns of typing the password
	1) clustering of letters : any possible combinations
3) Frequency decomposition : typing speed(different devices), correction speed, pointer speed
4) mental math speed captured from captcha
5) scrambled keypad to type the password
6) time taken for using a two factor code
7) device
8) flashing lights


Type 1 and type 2 errors

---

## Where each idea went

| # | Idea | Developed as |
|---|---|---|
| 1 | Trick the user: reaction and correction patterns | [[big_idea/08 Active Challenge Designs\|A1 Perturbation probe]] — note prior art US10298614B2 |
| 2 | Patterns of typing the password; clustering of letters | [[big_idea/08 Active Challenge Designs\|A2 Password motor-chunk structure]] — **the strongest idea in the set** |
| 3 | Frequency decomposition: typing, correction, pointer speed | [[big_idea/08 Active Challenge Designs\|A3 Frequency decomposition]] |
| 4 | Mental math speed from captcha | [[big_idea/08 Active Challenge Designs\|A4 Cognitive timing]] |
| 5 | Scrambled keypad to type the password | [[big_idea/08 Active Challenge Designs\|A5 Scrambled keypad]] — uniquely replay-proof |
| 6 | Time taken for using a two-factor code | [[big_idea/08 Active Challenge Designs\|A6 OTP entry timing]] — zero added friction |
| 3, 7 | Typing speed differs across devices; device | [[big_idea/08 Active Challenge Designs\|A7 Device-conditioned behaviour]] — templates must be per (user, device class). A correctness requirement, not an option |
| 8 | Flashing lights | [[big_idea/08 Active Challenge Designs\|A8a Visual reaction probe]] (person) and **A8b display refresh-rate fingerprint** (device). ⚠️ photosensitivity: stay under 3 flashes/sec per WCAG 2.3.1 — prefer a single non-repeating stimulus |
| — | Type 1 and type 2 errors | [[big_idea/06 Frameworks and Validation\|Frameworks and validation]] — FRR is Type I, FAR is Type II; base rates make them wildly asymmetric |
