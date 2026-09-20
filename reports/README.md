# BioPrint research reports

The **20 September 2026** snapshot contains two reports:

- [Dataset results and methodology](2026-09-20/DATASET_RESULTS.md): before/after FAR, FRR and EER; successful and negative experiments; paired app experiments; mathematical interpretation; all 72 recorded DEV rows.
- [Login integration priorities](2026-09-20/LOGIN_INTEGRATION_PRIORITIES.md): high-impact findings, compatibility constraints, four candidate combinations and a ten-hour plan prioritizing low impostor acceptance with step-up verification.

The snapshot includes source result JSON, precise CSV and SHA-256 checksums. It contains no production database or raw participant recordings. These reports do not enable research models in normal login.

To regenerate from the original local research artifacts:

```bash
bash scripts/research.sh reports/build_research_reports.py
```

The source artifacts and research Python wrapper are required for regeneration. The committed reports and evidence can be read independently.
