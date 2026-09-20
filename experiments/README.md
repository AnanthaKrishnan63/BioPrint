# Behavioral login experiments

The final typing-pointer implementation is merged into `main`. Run the app with
`python launch.py` from the repository root (or `bash experiments/run.sh`). Both
use localhost:8000 and bundled model assets; no dataset is required for login.

For optional dataset downloads, synthetic-cohort reconstruction, historical
branches and the trained-typing demo, see [docs/DATASETS.md](../docs/DATASETS.md).

- [Original four-policy results](RESULTS.md)
- [Trained-pointer protocol](NEURAL_POINTER_PROTOCOL.md)
- [Trained-pointer results](NEURAL_POINTER_RESULTS.md)
- [One-command typing demonstration](typing_demo/README.md)

The historical shared cohort has 39 artificial profiles / 390 scenarios: 12 TRAIN,
12 calibration and 15 DEV. Typing comes from public CMU recordings; the original
motor/device/trust signals are simulated. The neural follow-up adds the actual
trained pointer encoder on independently mapped public pointer recordings. These
mappings do not establish that participants across datasets are the same people.

Recorded reports under `results/` are historical evidence, not measurements of a
new user's installation. Final mobile policy adjustments postdate their aggregates.
