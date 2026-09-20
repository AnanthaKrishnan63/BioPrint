# Response to the draft critique

The revised six-page PDF replaces the initial draft. The first version is preserved in `versions/v1/`.

## Definitions and methodology

- Section 1 defines training-side data (TRAIN), development evaluation (DEV), enrollment support, step-up, false acceptance rate, false rejection rate, equal error rate and percentage points.
- Section 2 defines all timing and scoring symbols at their equations, including the typing score, threshold and ratio, the device-disagreement groups, and every term in the pointer cosine calculation.
- Radial basis function support vector machine (RBF SVM) is explained as a nonlinear owner-versus-other-user timing classifier. The [scikit-learn guide](https://scikit-learn.org/1.7/modules/svm.html) is cited in the PDF.
- Device fingerprinting now has a dedicated subsection: attributes collected, majority enrollment references, group weights, threshold, and its role in challenge routing. The researched FPStalker classifier is distinguished from the app's weighted attribute comparison.
- Table 1 explains what each main dataset contains and exactly what changed from baseline to candidate. Table 2 supplies the corresponding results.

## More data and learned typing

CMU's 100-enrollment SVM reaches EER 9.14%, compared with 14.32% for the ten-enrollment account-selected SVM; FRR is 44.43% versus 67.43% near 1% FAR. This supports richer personal reference data in these experiments. Different settings/support budgets mean it is not a fixed-model learning curve, and it does not establish improvement for every arbitrary password.

The proposed progressive strategy is now explicit: begin with a personal distance profile and use the existing learned path when a matching feature schema, labeled other-user background bank and calibration data exist. Repeated successful logins alone neither create that bank nor validate a new password model. No production scoring code was changed for this report or demo.

The new typing-only diagnostic uses saved SVM scores from the existing synthetic cohort: ordinary desktop scenarios give FAR 1/30 = 3.33%, FRR 24/45 = 53.33%. Including owner-like and scripted scenarios with available typing gives FAR 8/60 = 13.33%. These are explicitly labeled post-hoc, typing-only diagnostics rather than full-login or independently collected results.

## Pointer comparison

Pointer score mathematics and source counts were checked against the actual encoder adapter and saved API replay. At equal observation budget, source false accepts change from 145/1,748 to 29/1,748 and false rejects from 46/76 to 33/76. The paired synthetic application comparison is a different experiment: 14/90 to 9/90 false accepts; 13/60 to 19/60 false rejects.

Neither pointer policy is established as statistically superior overall. Exact account-level sign-flip tests give two-sided probability values 0.125 for FAR and 0.15625 for FRR. Repeated scenarios stay within the same account during all 32,768 label swaps. These are exploratory results conditional on the synthetic pairing and exchangeability assumption, not population confirmation. The observed FRR change is seven worsened and one improved genuine case, a net six additional failures.

Statistical pointer has 27 total errors versus 28 for trained pointer when errors have equal cost. On this fixed scenario mix, trained pointer becomes preferable when the cost of false acceptance exceeds 1.2 times the cost of false rejection. This supports a security-prioritized engineering choice, not a claim of universal statistical dominance.

## Additional experiments and conclusion

General typing now answers a specific question: can the compatible-password restriction be removed without unacceptable rejection? BEACON now answers whether independently promising keyboard and pointer encoders fuse successfully on the same people. Their data, fusion method and architectural implications are explained; the full variant ledger remains available.

The mobile-registration exclusion has been removed. The conclusion emphasizes the implemented contribution, measured improvements, and targeted additional evidence. It makes a strong case for the design without claiming that incompatible benchmarks prove superiority over all alternatives.

## One-command demonstration

```bash
bash experiments/typing_demo/run.sh
```

The script creates synthetic enrollments in a temporary database, fits the actual compatible SVM, calls the login API and verifies model scores and issued sessions. It opens a local browser report with four typing-classifier outcomes and one explicit owner-like attack. The ordinary model false accept is intercepted by the stricter application cutoff and shown as a keypad request, not mislabeled as a successful login. Examples are selected to explain behavior, not estimate accuracy. No live database or server is changed.
