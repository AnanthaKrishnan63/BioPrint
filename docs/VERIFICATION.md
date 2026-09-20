# Release verification

Verified on Linux x86_64 with Python 3.12, from a separate clean Git clone and a
fresh isolated environment installed from the pinned requirements:

- Dependency installation and `pip check`: passed.
- `python launch.py --check`: bundled checksum validation and CPU pointer inference passed.
- Public CMU download and synthetic-cohort reconstruction: original data hashes reproduced.
- Real-API typing demonstration: all five scores, access decisions and session outcomes passed.
- Application regression suite: **300 passed**. Optional research-artifact suites are separate.
- Occupied-port behavior: launcher refused to replace the existing localhost:8000 service.
- Final report: exactly six pages; build rejects page-count drift.

Application suite command (after installing `requirements-dev.txt`):

```bash
PYTHONPATH=code/bioprint:scripts OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python -m pytest code/bioprint/tests \
  --ignore-glob='*test_research*' --ignore-glob='*test_type2branch*' \
  --ignore-glob='*test_cmu_far*' --ignore-glob='*test_strict*' -q
```

These checks establish installation and functional behavior, not population
biometric accuracy. Research reproductions have their own dependencies, source
partitions and required artifacts. No live participant database was used.
