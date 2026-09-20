# LTMouseAuthen reproduction feasibility audit

Audit date: 2026-09-20. Outcome: **not sufficiently specified for a faithful implementation**. Existing FCN results remain unchanged. No new biometric measurements were read, models fitted, or dev scores generated during this audit.

## Primary evidence

Reviewed [arXiv v2, 11 May 2025](https://arxiv.org/html/2504.21415v2), including sections III, VI, VII and the actual [PDF Figure 5 on page 7](https://arxiv.org/pdf/2504.21415v2). The figure shows a pretraining residual network followed by transferred residual features, GRU, reshape and classification. Widths remain symbolic C1/C2/C3; numeric convolution widths/kernels and GRU configuration are absent. Adam betas and cross-entropy are specified, but learning rate, batch size, epoch count, initialization and freeze/fine-tune schedule are not. Section VII-D mentions consistent settings without supplying those values.

Input is scalar displacement magnitude divided by a presumed constant sampling interval. The interval and irregular-event resampling policy are unspecified. The paper describes per-session processing and data-dependent segment lengths; its Balabit range is 90–130. Existing 128-event displacement blocks cannot silently substitute for this velocity contract. The paper also uses claimant-specific binary training and sampled negatives, unlike the existing transferable FCN encoder.

## Code discovery and decision

Exact model-name, paper-title and arXiv-ID web searches, including GitHub-restricted searches, did not locate an attributable author implementation. The public [GitHub repository metadata query](https://api.github.com/search/repositories?q=LTMouseAuthen) returned zero repositories with `incomplete_results=false`. This is evidence of an unsuccessful search, not proof that code does not exist.

The missing dimensions and training schedule prevent a defensible CPU runtime estimate. Choosing plausible settings would create an independent architecture rather than reproduce this model. Therefore no training run was launched. Resumption requires attributable implementation/configuration or author clarification, followed by a frozen training-only protocol. No author was contacted. The established SapiMouse FCN baseline must not be described as a reproduction of this 2025 SOTA claim.
