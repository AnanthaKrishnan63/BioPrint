---
title: Basics and Glossary — Start Here
aliases:
  - Cheatsheet
  - Acronyms
tags:
  - big-idea
  - identity
  - research/primer
updated: 2026-09-18
---

# Basics and glossary — start here

Everything needed to read the rest of [[big_idea/00 Home|this folder]] without prior background. No prerequisites assumed beyond knowing what a web page and a password are.

> [!summary] If you read only one box
> The whole field is about **one decision**: when someone logs in, do we believe they are who they claim to be, or do we ask for more proof? We never get certainty. We get a **score**, we pick a **threshold**, and the threshold trades off two kinds of mistake: annoying real users versus letting attackers through. Almost every technical term below is either **a signal that feeds the score**, or **a way of measuring how bad the two mistakes are**.

---

## Part 1 — The problem in plain language

Someone types the right password into your site. The password is correct. **Is it the actual owner?**

It might not be. The password could have been stolen (phishing, a data breach, malware), or the owner might have left their laptop unlocked and a housemate sat down, or someone might be secretly controlling their computer from elsewhere.

You cannot see the person. All you have is what their **web browser** tells you. So the question becomes: *from browser data alone, can we tell the owner apart from someone else?*

There turn out to be two very different sub-questions, and confusing them is the single most common mistake:

1. **Is this the owner's usual computer?** Answerable by examining the machine — its screen size, graphics card, installed fonts, and so on. This is called **fingerprinting**.
2. **Is this the owner's hands?** Answerable only by examining *how* the person types and moves the mouse. This is called **behavioural biometrics**.

They fail in opposite situations:

- Owner buys a new laptop → the machine looks wrong, the hands look right. **Don't block them.**
- Housemate uses the owner's unlocked laptop → the machine looks perfect, the hands look wrong. **This is the dangerous one, and fingerprinting is completely blind to it.**

That is why the research keeps insisting on **two separate axes**. See the 2×2 table in [[big_idea/00 Home]].

---

## Part 2 — Identity and authentication basics

### The four words people mix up

| Term | Meaning | Everyday analogy |
|---|---|---|
| **Identification** | *Who are you?* — picking one person out of many (1-to-many) | A face-recognition camera scanning a crowd for anyone it knows |
| **Verification** / **authentication** | *Are you who you claim to be?* — checking one claim (1-to-1) | Showing your passport at a border: they compare **you** to **that one photo** |
| **Authorisation** | *Are you allowed to do this?* — happens **after** authentication | Having a valid ticket, but only for economy class |
| **Assurance** | *How confident are we?* — a level, not a yes/no | "Reasonably sure" vs. "certain" |

This project is **verification**, not identification. That is good news: 1-to-1 is a far easier problem than 1-to-many.

### Authentication factors

Classically three kinds of evidence:

| Factor | Means | Examples |
|---|---|---|
| Something you **know** | A secret in your head | Password, PIN |
| Something you **have** | A physical object | Phone, security key, the specific laptop |
| Something you **are** | A body property | Fingerprint, face, **how you type** |

**MFA / 2FA** (multi-/two-factor authentication) = requiring evidence from **two different categories**. A password plus a code texted to your phone is know + have. A password plus a second password is *not* MFA — both are "know".

Everything in this research is used to strengthen or trigger these, not replace them.

### Two kinds of biometrics

| Type | What it measures | Examples | Changes over time? |
|---|---|---|---|
| **Physiological** | Body structure | Fingerprint, iris, face | Barely |
| **Behavioural** | How you *do* things | Typing rhythm, mouse movement, gait, signature | **Yes, constantly** |

This project uses **behavioural** biometrics only, because a browser cannot read your fingerprint but it can time your keystrokes. Behavioural is weaker and noisier than physiological — but it is the only kind that can run **continuously and invisibly** while you work.

### Enrolment, template, verification

The three-step lifecycle of every biometric system:

1. **Enrolment** — collect samples from the known-genuine user (e.g. they type their password 20 times).
2. **Template** — the stored summary of those samples. *Not* the raw data: usually an average and a spread, or a compact numeric vector.
3. **Verification** — a new sample arrives; measure how far it is from the template; if close enough, accept.

> [!note] Why "template, not raw data" matters
> Storing raw keystroke timings is both a privacy liability and legally risky. Storing a derived template is safer and usually just as effective. See the legal section in [[big_idea/06 Frameworks and Validation]].

### Cold start

A new user has no template yet, so behavioural checks cannot work on their first login. This is the **cold-start problem**, and it is why behavioural biometrics can never be the *only* mechanism.

### Session, cookie, session hijacking

- A **session** is the period you stay logged in. The server remembers you via a **cookie** — a small token your browser sends with every request.
- **Session hijacking** = stealing that cookie and using it. The thief skips the password entirely.
- This is why **binding the cookie to the device's hardware** (see DBSC below) is such a big deal: a stolen cookie becomes useless anywhere else.

### Risk-based / adaptive authentication, and step-up

- **RBA (risk-based authentication)**: compute a risk score from context and behaviour, and only demand extra proof when the score is bad. Most logins stay frictionless.
- **Step-up authentication**: the extra proof you demand — an OTP, a passkey tap, a challenge.
- **Friction**: industry word for "effort imposed on the user". The entire design game is *maximum security per unit of friction*.

---

## Part 3 — Measuring whether it works ★

**The most important section.** Get this and most of the research becomes readable.

### The two mistakes

The system's null assumption is *"this is the genuine owner."* Two ways to be wrong:

| | Truth: genuine owner | Truth: impostor |
|---|---|---|
| **System accepts** | ✅ correct | **False accept** — a miss. The *security* failure. |
| **System rejects / challenges** | **False reject** — a false alarm. The *usability* failure. | ✅ correct |

- **FAR — False Acceptance Rate** = of all impostor attempts, the fraction wrongly let in. *Security cost.*
- **FRR — False Rejection Rate** = of all genuine attempts, the fraction wrongly challenged or blocked. *Usability cost.*

In classical statistics: a **false reject is a Type I error** (false positive alarm), a **false accept is a Type II error** (a miss).

### The threshold, and why there is no free lunch

The system produces a **score**; you choose a cutoff.

```
Loosen the threshold  → fewer genuine users annoyed (FRR ↓) but more impostors let in (FAR ↑)
Tighten the threshold → fewer impostors let in (FAR ↓) but more genuine users annoyed (FRR ↑)
```

You cannot reduce both by moving the threshold. You can only reduce both by building a **better system**. This is the central trade-off of the entire field.

### EER, ROC, AUC

- **EER — Equal Error Rate**: the single point where FAR = FRR. One number that summarises a system, so papers love it. **Lower is better.** EER of 5% means: at the balanced setting, 5% of impostors get in and 5% of real users get hassled.
- **ROC curve — Receiver Operating Characteristic**: the full picture. Plot FAR against FRR (or true-accept against false-accept) as the threshold sweeps from loose to tight. Shows performance at *every* possible setting, not just the balanced one.
- **AUC — Area Under the Curve**: the area under the ROC. **1.0 = perfect, 0.5 = coin flip.** A threshold-free summary.

> [!warning] EER is a comparison number, not an operating point
> EER assumes the two mistakes cost the same. In real life they do not — see the next box. Papers report EER so results are comparable; **systems should never actually run at the EER threshold.** This is why the research notes insist on reporting "FRR at FAR = 0.1%" instead: fix the security cost at something acceptable, then report how much user pain that costs.

### Base rate — why good-looking numbers fail in production ★

The crucial idea, and the one beginners find most surprising.

Suppose a site gets **1,000,000 logins a day**, and genuine account-takeover attempts are **1 in 10,000** — so about **100 impostors** and **999,900 real users** per day.

Run the system at **FRR = 1%** (sounds excellent!):

```
False alarms (real users challenged):   999,900 × 1%  ≈ 9,999
True detections (impostors caught):     about 99
Ratio: roughly 100 false alarms for every 1 genuine catch
```

**Every security team's alert queue drowns.** Tighten to FRR = 0.1% and you get ~10 false alarms per catch; at 0.01%, ~1 per catch — but tightening FRR usually means loosening FAR, so you start missing impostors.

Two consequences that shape the whole design:

1. **Report FRR at a low fixed FAR**, not EER. A 5% EER sounds fine and is unusable at this base rate.
2. **Never respond to a low score by blocking.** Respond with *friction* — ask for an OTP or a passkey tap. Then a false alarm costs a real user four seconds instead of their account. This is why [[big_idea/06 Frameworks and Validation]] uses **decision bands** rather than accept/reject.

### Detection delay

For **continuous** authentication (checking constantly during a session, not just at login), accuracy is not enough — **how long until you notice?** A system with a great EER that needs 20 minutes of typing is useless if the intruder finishes in three. Detection delay is measured in keystrokes or seconds.

---

## Part 4 — Information and uniqueness (entropy)

### Bits, intuitively

**Entropy**, measured in **bits**, answers: *how much does learning this fact narrow down who you are?*

**One bit halves the population.**

- Learning someone's **screen resolution** might cut 8 billion people to 500 million → about 4 bits.
- Learning their **exact font list** might cut that to a few thousand → another 10–14 bits.
- Bits from independent facts **add up**.

Since 2³³ ≈ 8.5 billion, **about 33 bits uniquely identifies one person on Earth.** A full browser fingerprint yields **50+ bits** — far more than needed. That surplus is why the notes say *stop maximising entropy and start maximising stability instead*: more signals mean more things that can change when the browser updates, causing false "new device" alarms.

**Shannon entropy** is the formula behind this: `H = −Σ p·log₂(p)`. You do not need to derive it; you need to know that it turns "how varied is this attribute across people" into a number of bits.

### Anonymity set

The group of people who share your exact value. Your screen resolution puts you in an anonymity set of millions; your full fingerprint usually puts you in a set of **one**. Small anonymity set = highly identifiable.

### Fingerprint, passive vs. active

- **Passive signals**: arrive without you asking (the browser sends them automatically — headers, network characteristics).
- **Active signals**: you must run code to collect them (draw on a canvas, probe for fonts, time an audio operation).

### Hashing vs. fuzzy matching

- **Hash**: squash many values into one short string. Change *one bit* of input and the hash changes completely.
- **Why that's wrong here**: browsers update roughly monthly and change their graphics output, version strings and GPU names. Hashing means every update looks like a brand-new device.
- **Fuzzy matching**: keep the individual attributes and compute a *similarity score*. 85% match = same device, drifted slightly. This is what [[big_idea/07 Browser Signal Catalogue]] means by "never hash all attributes into one ID".

### Drift

The slow change of a signal over time — browser updates changing the fingerprint, or a person's typing changing because they bought a new keyboard, got injured, or are simply tired. **Template adaptation** is the fix: gently update the stored template after confirmed-genuine sessions.

---

## Part 5 — Web and browser basics

### How a login reaches the server

```
Your browser ──[ TLS handshake ]──► Server        ← reveals the TLS fingerprint (JA3/JA4)
            ──[ HTTP request  ]──►                 ← reveals headers, their order, HTTP/2 settings
            ──[ page loads    ]──►
            ◄─[ JavaScript runs in the page ]      ← collects canvas, fonts, screen, keystroke timings
            ──[ results POSTed back ]──►
```

Two collection points, with very different properties:

| | Client-side (JavaScript) | Server-side (network) |
|---|---|---|
| Examples | Canvas, fonts, screen, keystrokes, mouse | TLS fingerprint, header order, IP address |
| Attacker can fake it? | **Yes** — it runs on their machine | **Very hard** — it is how their software actually speaks |
| Availability | Needs JS enabled | Always |

This is why the notes emphasise **consistency checks**: if the JavaScript claims "Chrome on macOS" but the TLS handshake looks like a Python script, that is a hard fail — and the attacker cannot fix it from inside the page.

### Terms you'll hit

| Term | Meaning |
|---|---|
| **API** (Application Programming Interface) | A function the browser offers to page code. `navigator.hardwareConcurrency` is an API that reports CPU core count. |
| **DOM** (Document Object Model) | The browser's live model of the page — the structure JavaScript reads and edits. |
| **Event** | Something that happened: `keydown`, `keyup`, `pointermove`, `click`. Each carries a **timestamp**. |
| **`keydown` / `keyup`** | Key pressed / released. **Dwell time** = keyup − keydown for one key. **Flight time** = keyup of one key → keydown of the next. |
| **`event.code` vs `event.key`** | `code` = *which physical key* (`KeyA`). `key` = *which character* (`a` or `A`). **Always log `code`, never `key`** — timing without characters is not a keylogger. |
| **`localStorage` / IndexedDB** | Browser storage that persists between visits. Deleted by clearing browser data or using private mode. |
| **Cookie** | Small token sent with every request; how the server remembers your session. |
| **Same-origin policy** | A page can only read data from its own site. This is why cross-site tracking is hard and getting harder. |
| **Secure context** | The page is served over HTTPS. Many sensitive APIs refuse to work otherwise. |
| **Permission prompt** | The browser asks the user before granting camera, microphone, motion sensors, etc. Anything behind a prompt is unusable for *silent* checks. |
| **Headless browser** | A browser with no visible window, driven by a script. The standard attacker tool. |
| **Canvas** | A drawing surface. Drawing text and reading the pixels back reveals tiny rendering differences between graphics stacks — **the highest-entropy single signal**. |
| **IP address** | Your network address. Roughly locates you and identifies your provider. |
| **ASN** (Autonomous System Number) | Identifies the *network operator* — e.g. Airtel, Jio, or Amazon's data centres. More stable and more useful than the IP itself. |
| **Geo-IP** | Estimating physical location from IP. City-level at best, often wrong on mobile networks. |
| **VPN / proxy / Tor** | Tools that route traffic elsewhere, hiding the real IP. Common and legitimate — never treat as proof of guilt. |
| **Impossible travel** | Two logins so far apart in space and so close in time that no aircraft could connect them. Classic risk signal. |
| **RAT** (Remote Access Trojan) | Malware letting an attacker control the victim's actual computer. **The fingerprint is perfect and genuine** — only behaviour gives it away. |

### Timer coarsening — a quirk that matters a lot here

Browsers deliberately make their clocks **less precise** (~100 microseconds in Chrome, 1 millisecond in Firefox) to block a class of CPU attacks called **Spectre**. Since keystroke dwell times are only tens of milliseconds, this rounding **blurs exactly the measurement this research depends on**. Several notes call for testing features at 1 ms resolution rather than assuming full precision.

---

## Part 6 — Machine learning basics (only what's needed)

| Term | Meaning |
|---|---|
| **Feature** | One measured number. "Dwell time on the `T` key" is a feature. |
| **Feature vector** | All features for one sample, as a list of numbers. The notes mention a "28-dimensional vector" = 28 measurements from one password entry. |
| **Classifier** | Something that takes a feature vector and outputs a label or score. |
| **Training / test split** | Fit the model on one part of the data, measure it on unseen data. **Measuring on data you trained on always flatters you.** |
| **Data leakage** | Accidentally letting test information into training. The commonest cause of unbelievably good published results. |
| **Overfitting** | Memorising the training data instead of learning the pattern. Great on training data, useless on new data. |
| **Distance metric** | How far apart two vectors are. **Manhattan** = sum of absolute differences. **Euclidean** = straight-line. **Scaled/Mahalanobis** = distance that accounts for how much each feature naturally varies — a 5 ms deviation matters more on a steady feature than on an erratic one. |
| **One-class classification** | Training when you only have examples of *one* class. Here: you have the real user's typing but no impostor's. Methods: **one-class SVM**, **isolation forest**, distance-to-mean. |
| **Global impostor pool** | The practical workaround: use *other users'* data as stand-in impostors. Usually beats true one-class methods. |
| **Embedding** | A learned compact vector representing a sample, where similar samples land near each other. |
| **Siamese / triplet network** | A network trained on *pairs* or *triples* — "these two samples are the same person, this third is not". Produces embeddings. **Advantage: works for new users without retraining.** |
| **Logistic regression** | The simplest useful classifier. Weighted sum of features → probability. Fast, interpretable, a strong baseline. |
| **LLR** (Log-Likelihood Ratio) | `log( P(evidence if genuine) / P(evidence if impostor) )`. Positive = evidence for genuine. **LLRs from independent signals simply add up**, which is why [[big_idea/06 Frameworks and Validation]] fuses scores this way — a missing signal just contributes 0 instead of breaking the model. |
| **Federated learning** | Training across many devices without collecting the raw data centrally. The privacy-friendly option for behavioural biometrics. |
| **Transformer / CNN / ViT** | Neural architectures. **CNN** for grid-like data, **Transformer** for sequences, **ViT** (Vision Transformer) applies transformers to images. Papers use these to classify mouse trajectories rendered as pictures. |

> [!note] Do not skip the simple baseline
> On the standard keystroke benchmark, **scaled Manhattan distance** — barely more than "how far is this from the average, weighted by usual variability" — beats most deep-learning approaches published since. Build it first and make anything fancier prove it earns its place.

---

## Part 7 — Signal processing basics (only what's needed)

Needed only for [[big_idea/08 Active Challenge Designs|design A3]].

| Term | Meaning |
|---|---|
| **Time series** | Measurements in time order — e.g. mouse speed every 8 ms. |
| **Sampling rate (Hz)** | Measurements per second. 125 Hz = 125 per second. |
| **Frequency domain** | Instead of "what happened when", ask "what *rhythms* are present". A chord in the time domain is a waveform; in the frequency domain it is three peaks. |
| **FFT** (Fast Fourier Transform) | The standard algorithm for converting to the frequency domain. |
| **PSD** (Power Spectral Density) | How much energy sits at each frequency. **Welch's method** is the usual robust way to estimate it. |
| **Spectral centroid** | The "centre of mass" of the spectrum — roughly, how high-pitched the signal is. |
| **Physiological tremor** | Everyone's hand shakes slightly at **8–12 Hz**. It is involuntary, individually characteristic, and shows up as a bump in the mouse-velocity spectrum. **You cannot decide to change it** — which is what makes it attractive as a biometric. |
| **Wavelet transform** | Like FFT but localised in time; better when the signal's character changes as it goes. |

**Why bother with frequency at all?** Time-domain features like "average speed" change when someone is in a hurry. Frequency-domain features like "where is the tremor peak" stay put. They are more **invariant to how fast the person happens to be working**, which is one of the largest sources of noise in behavioural biometrics.

---

## Part 8 — Human factors basics

| Term | Meaning |
|---|---|
| **Fitts's law** | Pointing takes longer when the target is far away or small: `MT = a + b·log₂(D/W + 1)`. MT = movement time, D = distance, W = target width. **`a` and `b` are personal constants** — fit them per user and you get a compact two-number signature of how that person points. |
| **Hick's law** | Decision time grows with the number of options. Used in the choice-reaction challenge. |
| **Reaction time (RT)** | Stimulus → response. Typically 200–250 ms. Varies more *between* people than *within* one person, which is what makes it usable. |
| **Ex-Gaussian** | The distribution RTs actually follow: a bell curve plus a long right tail. Three parameters — μ, σ, **τ** (the tail). **τ is the personal one**; average RT alone is a weak feature. |
| **Motor chunking** | Practised sequences get stored as **units**, not individual moves. A password you've typed 5,000 times is executed as ~3 chunks, not 10 keystrokes. Chunk boundaries appear as **pauses** in the timing. |
| **Why chunking is the strongest idea here** | Someone who *knows* your password still types it like a beginner — evenly paced, no chunk structure. **Knowing the secret doesn't give you the motor program.** See [[big_idea/08 Active Challenge Designs\|design A2]]. |
| **Dwell / flight time** | Key hold duration / gap between keys. The two atoms of keystroke dynamics. |
| **Digraph / trigraph** | A pair / triple of consecutive characters. "Digraph latency" = time between pressing `t` and `h` in "the". |
| **Sensorimotor feedback control** | The involuntary loop that corrects your hand mid-reach when something moves. Much harder to fake than typing rhythm, which is why perturbation probes work. |
| **Photosensitive epilepsy / WCAG 2.3.1** | Flashing visuals can trigger seizures. The accessibility standard forbids more than **three flashes per second**. A hard constraint on any "flashing lights" design. |

---

## Part 9 — Acronym dictionary

| Acronym | Expansion | One-line meaning |
|---|---|---|
| **AAL** | Authentication Assurance Level | NIST's 1/2/3 scale of how confident an authentication method is |
| **API** | Application Programming Interface | A function the browser exposes to page code |
| **ASN** | Autonomous System Number | Identifies the network operator behind an IP |
| **ATO** | Account Takeover | An attacker gaining control of someone's account — the threat this research addresses |
| **AUC** | Area Under the Curve | Threshold-free score for a classifier; 1.0 perfect, 0.5 random |
| **BIPA** | Biometric Information Privacy Act (Illinois) | US state law with real damages for mishandling biometric data |
| **CDP** | Chrome DevTools Protocol | The channel automation tools use to drive Chrome; leaves detectable traces |
| **CHIPS** | Cookies Having Independent Partitioned State | Cookies isolated per top-level site |
| **CNN** | Convolutional Neural Network | Neural net suited to grid-shaped data like images |
| **DBSC** | Device Bound Session Credentials | Chrome feature binding a session cookie to the device's security chip — **a stolen cookie becomes useless elsewhere** |
| **DPDP** | Digital Personal Data Protection Act, 2023 | India's data protection law |
| **EER** | Equal Error Rate | The threshold where FAR = FRR; a one-number summary. Lower is better |
| **FAR** | False Acceptance Rate | Fraction of impostors wrongly let in — the security cost |
| **FFT** | Fast Fourier Transform | Converts a time signal to its frequency content |
| **FIDO** | Fast IDentity Online | The alliance behind WebAuthn and passkeys |
| **FRR** | False Rejection Rate | Fraction of genuine users wrongly challenged — the usability cost |
| **FTO** | Freedom to Operate | A legal check that you don't infringe existing patents |
| **GDPR** | General Data Protection Regulation | EU privacy law; **Article 9** covers biometric data specifically |
| **HCI** | Human–Computer Interaction | The field Fitts's law and reaction-time work come from |
| **JA3 / JA4** | *(not an acronym — tool names)* | Fingerprints of how a client performs the TLS handshake. Server-side, **unspoofable from page JavaScript** |
| **LLR** | Log-Likelihood Ratio | Evidence strength on a scale where independent pieces add |
| **MFA / 2FA** | Multi- / Two-Factor Authentication | Requiring evidence from two different factor categories |
| **NIST** | National Institute of Standards and Technology (US) | Publishes **SP 800-63**, the reference standard for digital identity |
| **OTP** | One-Time Password | The short code sent by SMS or generated by an app |
| **PSD** | Power Spectral Density | Energy distribution across frequencies |
| **RAT** | Remote Access Trojan | Malware letting an attacker drive the victim's own machine |
| **RBA** | Risk-Based Authentication | Score the risk; demand extra proof only when it's high |
| **RFP** | Resist Fingerprinting | Firefox's hardening mode; spoofs or blocks many signals |
| **ROC** | Receiver Operating Characteristic | The curve showing the full FAR/FRR trade-off |
| **RT** | Reaction Time | Stimulus-to-response latency |
| **SoK** | Systematisation of Knowledge | A survey-style paper that organises a whole field |
| **SP 800-63** | NIST Special Publication 800-63 | The digital identity guidelines; **-63B** covers authentication |
| **SVM** | Support Vector Machine | A classical classifier; the one-class variant suits this problem |
| **TLS** | Transport Layer Security | The encryption behind HTTPS; its handshake is fingerprintable |
| **TPM** | Trusted Platform Module | Security chip that stores keys so they **cannot be copied off the device** |
| **UA** | User-Agent | The string a browser sends identifying itself. Chrome has deliberately **frozen** it to reduce tracking |
| **UA-CH** | User-Agent Client Hints | The replacement: the site must explicitly *ask* for the detail the UA string used to volunteer |
| **ViT** | Vision Transformer | Transformer architecture applied to images |
| **W3C** | World Wide Web Consortium | Standards body for web APIs |
| **WCAG** | Web Content Accessibility Guidelines | Accessibility standard; **2.3.1** is the three-flashes-per-second limit |
| **WebAuthn** | Web Authentication (W3C standard) | The browser API behind **passkeys** — cryptographic login with no shared secret to steal |
| **WebGL / WebGPU** | Web Graphics Library / GPU | Graphics APIs; they expose the GPU's identity, making them strong fingerprint signals |

---

## Part 10 — How to read a paper in this field

Before believing **any** reported accuracy, extract these four things. If a paper omits them, the number is not comparable to anything.

| Question | Why it matters | Red flag |
|---|---|---|
| **1. Same-session or cross-session?** | Training and testing on one sitting inflates results 2–5× | Doesn't say → assume same-session |
| **2. How many subjects?** | 10 users is a demo, not evidence | Under ~20 |
| **3. How much data at test time?** | "99% accurate" on 1,000 keystrokes ≠ on 50 | Not stated |
| **4. Were the impostors seen during training?** | If yes, it learned those specific people, not "not-the-user" | Not stated |

> [!important] The rule of thumb that will save you the most time
> **A reported EER below 0.1% for behavioural biometrics almost always indicates same-session evaluation.** Realistic cross-session numbers are in the **3–15%** range. When [[big_idea/02 Papers]] flags a paper claiming EER = 0.000068 as "treat with scepticism", this is why.

---

## Part 11 — Beginner traps

| Trap | Reality |
|---|---|
| "Higher accuracy is better" | Not without the base rate. 99% accurate sounds great and is unusable when impostors are 1 in 10,000. |
| "More signals = better" | More signals = more things that change when the browser updates = more false alarms. **Stability beats entropy.** |
| "Just hash everything into one device ID" | One browser update and every user looks brand new. Fuzzy-match instead. |
| "Behavioural biometrics can replace passwords" | No. It's a **trigger** for asking for stronger proof, not proof itself. |
| "A VPN means they're an attacker" | VPN use is normal and rising. Weight *familiarity* (has this account used this network before?) over *category*. |
| "We can just watch how they type" | Only after enrolment, only on a device class you've seen them on before, and it's **legally special-category data**. |
| "Mobile works the same as desktop" | It inverts. Autocorrect and swipe typing destroy keystroke timing; phones are also less fingerprintable. Use motion sensors and touch instead. |
| "The lab number will hold up in production" | Cross-session, cross-device, cross-browser and real base rates each cost you. Assume a large gap. |
| "Fingerprinting the device tells us who the person is" | It tells you about the **machine**. The housemate at the unlocked laptop passes every fingerprint check perfectly. |

---

## Reading path

1. **This note** — the vocabulary.
2. [[big_idea/00 Home|00 Home]] — the 2×2 and the four evidence layers. The whole argument in one page.
3. [[big_idea/07 Browser Signal Catalogue|07 Browser Signal Catalogue]] — what can actually be measured. Skim the tables; don't memorise.
4. [[big_idea/06 Frameworks and Validation|06 Frameworks and Validation]] — how signals become a decision, and how to measure it honestly.
5. [[big_idea/08 Active Challenge Designs|08 Active Challenge Designs]] — the original ideas, developed.
6. [[big_idea/09 Experiment Plan|09 Experiment Plan]] — what to actually build.
7. [[big_idea/01 Landscape and Research Questions|01]], [[big_idea/02 Papers|02]], [[big_idea/03 Startups and Products|03]], [[big_idea/04 Patents|04]], [[big_idea/05 Datasets and Data Availability|05]] — reference material, read as needed.

**Two papers to read in full, in this order:**
- *Comparing Anomaly-Detection Algorithms for Keystroke Dynamics* (Killourhy & Maxion, 2009) — the clearest worked example of the whole measurement methodology.
- *What's in Score for Website Users* (2021) — what actually happens when this is deployed to real people.

## Related notes

- [[big_idea/00 Home|Big idea hub]]
- [[yooooooooo|Original brainstorm]]
