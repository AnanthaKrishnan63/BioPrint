# HMOG tap reference implementation contract

This supplement clarifies the frozen tap proposal before new tap extraction.
It does not authorize dataset reads or replace the original proposal.

## Enrollment and missing observations

The five-valid-taps minimum applies to enrollment windows as well as probe
windows. An account's 80-tap enrollment minimum counts only individual tap
vectors from enrollment windows that meet that five-tap minimum. This window
restriction is an explicit adaptation; the source's enrollment minimum alone
does not establish it. Profiles use every qualifying enrollment tap, with no
80-tap cap and no equal weighting of unequal-sized windows. Probe observations
never update a profile.

Authentication first averages valid tap vectors within a window, then measures
the scaled Manhattan distance to the account profile. It does not average
individual-tap distances. Zero-spread enrollment dimensions are ignored; an
account with no active dimensions has no verdict. All original windows and
account columns remain in coverage denominators.

## Comparisons still requiring a frozen experiment

The implemented reference uses all eleven dimensions. The proposal's optional
fixed three-feature ablation is not implemented or evaluated. If implemented,
it must use only dimensions 0, 1, and 10 and independently establish whether
any of those dimensions has usable enrollment spread. Availability of another
dimension cannot make this ablation scorable.

The prospective fusion helper fits a positive mean-impostor-distance scale per
modality on an explicit TRAIN claim mask, then averages scaled distances with
equal weights. It requires all components for each claim. Missing components
produce no verdict, never partial fusion. Identical window and account ordering
must be verified by the experiment runner; array shapes alone cannot prove
genuine pairing. The helper does not authorize fitting or establish provenance.

Before any comparison, freeze modality choices, scale-fit sessions, selection
and calibration roles, original and intersection coverage reports, and the
validation recipe. Previously exposed encoder DEV observations cannot become a
pristine validation set for this follow-up. Test observations remain sealed.
