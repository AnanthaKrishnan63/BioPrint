# Optional research data

**The login app needs no downloaded dataset.** Its pointer encoder and small typing
calibration bank are included in `models/`. Personal enrollment is collected locally.

Activate `bigidea` before running these commands. Research code and saved evidence
are included; large datasets, intermediate feature matrices and optional research
checkpoints are excluded. Those experiments may need preprocessing/training as well
as a download. Their dependencies are separate from the app runtime; see
`research/benchmarks/requirements-research.txt` and each script's help/source.
Some historical Type2Branch studies require a separate TensorFlow environment.

## Dataset helper

```bash
python scripts/datasets.py cmu                 # show source and expected location
python scripts/datasets.py cmu --download      # fetch and verify the frozen source
python scripts/datasets.py typing-demo --download  # also reproduce the synthetic cohort
python scripts/datasets.py sapimouse --download
python scripts/datasets.py balabit --download
python scripts/datasets.py device --download
```

Pointer downloaders need `curl` (`sudo apt install curl` on Ubuntu/Debian).
Downloads are explicit, remain local and follow each source's terms. Existing CMU
files are checksum-verified rather than replaced. Historical pointer downloaders
preserve sealed-test split rules. The dataset helper prints manual instructions
instead of pretending an unsupported source was installed.

| Dataset | Source / required placement | Preparation |
|---|---|---|
| CMU typing | [Author site](https://www.cs.cmu.edu/~keystroke/); `code/bioprint/eval/data/DSL-StrongPasswordData.csv` | Helper verifies the source SHA-256; strict readers omit sealed sessions before numeric parsing. |
| SapiMouse | [Author site](https://www.ms.sapientia.ro/~manyi/sapimouse/); `data/benchmarks/sapimouse/` | Helper runs `scripts/pointer_sapimouse_prepare.py`; retraining is unnecessary for login. |
| Balabit | [Author repository](https://github.com/balabit/Mouse-Dynamics-Challenge); `data/benchmarks/balabit/` | Helper fetches the assigned training/development sessions. |
| FPStalker | [Author repository](https://github.com/Spirals-Team/FPStalker); `datasets/fpstalker/` | Helper downloads checksum-pinned archives; evaluation uses the device preparation scripts. |
| KeyRecs | [Zenodo 7886743](https://doi.org/10.5281/zenodo.7886743); `research/benchmarks/data/keyrecs/` | Download `fixed-text.csv`, `free-text.csv` and save the record metadata as `source-metadata.json`; retain upstream CC BY 4.0 attribution. |
| BEACON | [Author release](https://huggingface.co/datasets/beacon-gui/BEACON-Dataset); `datasets/beacon/` | Obtain `public_release_manifest.csv` from revision `da1306428ef626108a914abff42ead19b3a46f62`, then run `python scripts/beacon_download.py --download`. It uses a bounded subset, not the full release. |
| HMOG | [Author site](https://www.cs.wm.edu/~qyang/hmog.html); `datasets/hmog/` | Follow source access terms. Historical acquisition additionally requires ZIP metadata under `datasets/joint_feasibility_metadata/`; see `research/benchmarks/hmog/README.md`. |
| Stroop/Flanker | [OSF source](https://osf.io/cwzds/) | `scripts/cognitive_benchmark.py` contains acquisition, source metadata and split preparation. |
| Touch TSI / DELBOT | Sources and expected layouts in `scripts/touch_benchmark.py` and `scripts/delbot_benchmark.py` | Inspect their preparation steps before running; these studies are not app dependencies. |
| Arithmetic / BrainRun | `research/benchmarks/arithmetic_sources.md`; `scripts/brainrun_acquire.py` | Feasibility studies require source metadata and bounded acquisition; they are not integrated login models. |

## Reproduce the typing demonstration

```bash
python -m pip install -r requirements-dev.txt
bash experiments/typing_demo/run.sh --no-open
```

On first run this downloads CMU and reconstructs the shared synthetic cohort.
It then verifies the cohort hashes, fits real per-account models through the API,
and writes `experiments/runtime/typing-demo/index.html` plus JSON results.
The temporary account database is deleted. These outcome-selected examples are
illustrations, not an unbiased accuracy benchmark.

## Experiments and branches

The four original policies used the same cohort. Their branches remain available
as `experiment/login-control`, `experiment/typing-stepup`,
`experiment/typing-pointer`, and `experiment/typing-keypad`. Additional research is
on `experiment/general-typing` and `experiment/research-model-fusion`.

A default clone checks out `main`, not our old worktree layout. Historical branches
preserve their original launch assumptions; use the root launcher on `main` for the
final app. For historical replay use an activated environment, a regenerated cohort,
and an explicit `BIOPRINT_RESEARCH_ROOT` pointing at the checkout containing that
cohort. Do not mix participant databases between branches.

Saved results and protocols are under `reports/` and `experiments/results/`.
Missing optional artifacts do not imply that a historical result was rerun. The
six-page report can be read without any research installation or data download.
