# Type2Branch CPU runtime candidate

All dependencies are local to `.research-type2branch-deps`; the bigidea Python
interpreter remains in use. This directory is a dependency-precedence overlay,
not a clean virtual environment. Pip reports and `package_inventory.json`
record39installed package versions and archive hashes. Legacy Keras is selected
before TensorFlow import. Keras/cache paths point inside the repository.

On a fresh checkout without these output directories, run:

```bash
bash scripts/research.sh scripts/type2branch_runtime_setup.py
bash scripts/research.sh scripts/type2branch_keras_compat.py
bash scripts/research.sh scripts/type2branch_reference_smoke.py
```

Scripts refuse to overwrite existing run directories. The setup script now
includes the SciPy pin found necessary during execution; `setup_source.py`
preserves the original two-stage installer. Actual supplemental installation
is recorded in `scipy_install.json` and `install_scipy.log`. The compatibility
script changes one integral float bound in legacy Keras; `keras_compat.json`
records before/after hashes and `keras_backend_original.py` preserves its input.

Successful architecture check: `../type2branch_reference_smoke_v3/report.json`.
Earlier attempts retain failure reports. Generated batch2 forward/backward with
a squared-output diagnostic loss is not Set2Set training, accuracy validation,
or a batch150/epoch performance estimate. No dataset files were read.
