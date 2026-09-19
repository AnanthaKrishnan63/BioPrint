---
title: Account Holder Verification — Papers
tags:
  - big-idea
  - identity
  - research/papers
updated: 2026-09-18
---

# Relevant papers

## Summary

The literature splits cleanly into three bodies that rarely cite each other: **web-tracking/fingerprinting** research (measurement-driven, privacy-motivated, and therefore describing these techniques as attacks), **behavioural-biometrics** research (ML-driven, benchmark-driven, with error rates that are frequently optimistic because of same-session evaluation), and **risk-based authentication** research (systems-driven, with very few large-scale studies because the data is proprietary). Reading across them is where the useful findings are.

> [!important] Finding to carry forward
> Published EERs below roughly 0.1% for behavioural biometrics are almost always **same-session** — train and test drawn from one sitting. Under honest cross-session, cross-device protocols, realistic numbers for free-text keystroke and mouse dynamics land in the **3–15%** range. Treat any vendor or paper figure without a stated evaluation protocol as uninformative.

## Fingerprinting and web tracking

| Paper | Finding summary | Evidence/use | Link |
|---|---|---|---|
| *SoK: Advances and Open Problems in Web Tracking* (2025, preprint) | Systematises the current tracking landscape including the post-Privacy-Sandbox state. The most current single overview of what still works after browser countermeasures. | Primary orientation text; use for the taxonomy and for open problems. | [arXiv 2506.14057](https://arxiv.org/pdf/2506.14057) |
| *Browser Fingerprinting: A Survey* (2019, preprint) | The canonical survey. Catalogues attributes, entropy contributions, and countermeasures; establishes the entropy/stability framing used throughout this vault. | Foundational reference for attribute selection and the entropy method. | [arXiv 1905.01051](https://arxiv.org/pdf/1905.01051) |
| Panopticlick / *How Unique Is Your Web Browser?* (Eckersley, 2010) | The original measurement showing most browsers are uniquely identifiable; introduced Shannon-entropy-per-attribute as the standard metric. | Historical anchor and the source of the methodology for [[big_idea/09 Experiment Plan|Experiment 1]]. | [EFF Panopticlick paper](https://coveryourtracks.eff.org/static/browser-uniqueness.pdf) |
| AmIUnique / *Beauty and the Beast* (Laperdrix et al., 2016) | Large-scale follow-up adding canvas and WebGL; quantifies mobile vs. desktop fingerprintability and shows mobile is *less* distinguishable. | Important caveat: the mobile case is materially harder for the device axis. | [AmIUnique](https://amiunique.org/) |

## Keystroke dynamics

| Paper                                                                                                                        | Finding summary                                                                                                                                                                                                         | Evidence/use                                                                                                         | Link                                                                                                           |                                           |
| ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| *Comparing Anomaly-Detection Algorithms for Keystroke Dynamics* (Killourhy & Maxion, DSN 2009)                               | The benchmark study. 51 subjects typing `.tie5Roanl` 400 times across 8 sessions; compares 14 detectors under one protocol. **Scaled Manhattan distance is the strongest, and it beats most later "advanced" methods.** | The baseline every fixed-text result should be compared against. Its dataset underpins [[big_idea/09 Experiment Plan | Experiment 2]].                                                                                                | [CMU](https://www.cs.cmu.edu/~keystroke/) |
| *Keystroke Dynamics: Concepts, Techniques, and Applications* (2023/2025, survey; ACM CSUR)                                   | Comprehensive modern survey covering fixed vs. free text, feature families, deep models, datasets and open problems.                                                                                                    | The single best entry point to the subfield.                                                                         | [arXiv 2303.04605](https://arxiv.org/pdf/2303.04605) · [ACM CSUR](https://dl.acm.org/doi/full/10.1145/3733103) |                                           |
| *Fast Free-text Authentication via Instance-based Keystroke Dynamics* (2020, preprint)                                       | Instance-based approach that reaches usable accuracy on far shorter free-text samples than prior work. Directly relevant to **detection delay**.                                                                        | Method candidate for continuous authentication with short windows.                                                   | [arXiv 2006.09337](https://arxiv.org/pdf/2006.09337)                                                           |                                           |
| *Free-Text Keystroke Dynamics for User Authentication* (2021, preprint)                                                      | Compares feature sets and classifiers on free text; documents the sample-size dependence explicitly.                                                                                                                    | Use for the EER-vs-window-size curve design.                                                                         | [arXiv 2107.07009](https://arxiv.org/pdf/2107.07009)                                                           |                                           |
| *Impact of Data Breadth and Depth on Siamese Neural Network Performance* (2025, preprint)                                    | Tests Siamese/embedding models across three public keystroke datasets; quantifies how many users (breadth) and how many samples per user (depth) are actually needed.                                                   | Directly informs how many subjects to recruit.                                                                       | [arXiv 2501.07600](https://arxiv.org/html/2501.07600)                                                          |                                           |
| *Client-Side Continuous Authentication Using Keystroke Dynamics: A Lightweight Pipeline and Cross-Session Evaluation* (2025) | Explicitly browser-side scoring with a **strict cross-session** protocol; digraph-timing features with per-user logistic-regression verifiers. Closest published work to this project's constraint.                     | The most directly transferable methodology. Read first.                                                              | [DOI 10.3390/electronics15112325](https://doi.org/10.3390/electronics15112325)                                 |                                           |
| *A Review of Several Keystroke Dynamics Methods* (2025, preprint)                                                            | Short comparative review; useful for quickly triaging method families.                                                                                                                                                  | Triage reading.                                                                                                      | [arXiv 2502.16177](https://arxiv.org/pdf/2502.16177)                                                           |                                           |

## Mouse and pointer dynamics

| Paper | Finding summary | Evidence/use | Link |
|---|---|---|---|
| *From Clicks to Security: Investigating Continuous Authentication via Mouse Dynamics* (2024, preprint) | Evaluates mouse-dynamics continuous authentication and is unusually explicit about protocol and its effect on reported performance. | Method and evaluation-design reference. | [arXiv 2403.03828](https://arxiv.org/pdf/2403.03828) |
| *Privacy-preserving and robust mouse dynamics authentication using hybrid transformer-CNN and federated learning* (2026) | Hybrid transformer-CNN with federated training; addresses the fact that raw behavioural data should not leave the device. | Architecture reference **and** a privacy-by-design pattern that matters for GDPR Art. 9 compliance. | [Frontiers in AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1914353/full) |
| Mazumdar & Sundaram (2025) — recurrence plots + modified ViT | Reports EER ≈ 0.0498 on the Balabit dataset by encoding mouse trajectories as recurrence plots and applying a vision transformer. | Strong method; note the dataset is small and the protocol favourable. | *(see survey below for citation context)* |
| Aljoubory & Mahdi (2025) — lightweight EfficientNet-inspired model | Reports 99.51% accuracy, EER ≈ 6.8×10⁻⁵. | **Treat with scepticism** — an EER this low on mouse dynamics is a strong indicator of same-session leakage. Useful as a cautionary example of protocol inflation. | *(see survey below)* |

## Fusion, continuous authentication, and surveys

| Paper | Finding summary | Evidence/use | Link |
|---|---|---|---|
| *Adaptability of current keystroke and mouse behavioral biometric systems: A survey* (2025) | Focuses squarely on the weak point: **template drift and adaptation over time**. Surveys how systems cope with users whose behaviour changes. | Essential for the template-update design. | [Computers & Security](https://www.sciencedirect.com/science/article/pii/S0167404825004201) |
| *User Authentication Method Based on Keystroke and Mouse Dynamics with Scene-Irrelated Features in Hybrid Scenes* (2022) | Proposes features that transfer across task contexts rather than being tied to one application screen. | Important — naive features overfit to the specific page layout used in training. | [PMC9460698](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460698/) |
| *Combining Mouse and Keystroke Dynamics Biometrics for Risk-Based Authentication in Web Environments* | Early, directly on-topic work fusing both modalities specifically for web RBA. | Historical anchor for the fusion approach. | [ResearchGate](https://www.researchgate.net/publication/261428602) |
| *Sensor-based Continuous Authentication of Smartphones' Users Using Behavioral Biometrics: A Contemporary Survey* (2020, preprint) | Covers accelerometer/gyroscope-based continuous auth — the mobile analogue, reachable in-browser via `devicemotion`. | Reference for the mobile branch, where keystroke timing is nearly useless. | [arXiv 2001.08578](https://arxiv.org/pdf/2001.08578) |
| *Enhancing security and usability with context aware multi-biometric fusion for continuous user authentication* (2025) | Context-aware fusion weighting — adjusts modality weights by situation rather than fixing them. | Directly relevant to the two-axis fusion design. | [PMC12368039](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12368039/) |

## Risk-based authentication in production

| Paper | Finding summary | Evidence/use | Link |
|---|---|---|---|
| *What's in Score for Website Users: A Data-driven Long-term Study on Risk-based Authentication Characteristics* (2021, preprint) | Rare longitudinal study on real RBA deployment data. Finds **IP address and ASN carry most of the practical signal**, quantifies how often legitimate users trip re-authentication, and examines usability cost. | The most grounded evidence on what RBA actually achieves in production. Read before choosing thresholds. | [arXiv 2101.10681](https://arxiv.org/pdf/2101.10681) |
| *NIST SP 800-63B-4, Digital Identity Guidelines: Authentication* (final, 2025) | Establishes AAL definitions, admits syncable passkeys at AAL2, and explicitly permits attestation and risk indicators (device swap, SIM change, abnormal behaviour) as inputs to verifier decisions. | The normative frame. Determines what claims the system may legitimately make. | [NIST SP 800-63B](https://pages.nist.gov/800-63-4/sp800-63b.html) |

## Synthesis for builders

> [!important] Finding
> The literature supports **risk scoring, step-up triggering, and session-anomaly detection**. It does **not** support a claim that browser-only signals can positively *identify* an individual with the reliability of a possession or biometric factor. The honest framing is: this system decides **when to ask for stronger proof**, and cryptographic binding provides that proof. Anything stronger overstates what the evidence will carry.

> [!warning] Evaluation protocol is the whole ballgame
> When reading any result in this area, extract four things before believing the number: (1) same-session or cross-session split, (2) number of subjects, (3) sample length at test time, (4) whether the impostor set was seen during training. Papers that omit these are not comparable to each other.

## Related notes

- [[big_idea/01 Landscape and Research Questions|Landscape and research questions]]
- [[big_idea/05 Datasets and Data Availability|Datasets and data availability]]
- [[big_idea/06 Frameworks and Validation|Frameworks and validation]]
- [[big_idea/09 Experiment Plan|Experiment plan]]
