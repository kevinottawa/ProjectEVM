# EVM MRI Domain Library

`v2/domain_library.json` is the canonical prompt and domain library for workload-pack discovery. It is organized hierarchically and separates prompts into calibration, validation, and held-out test splits.

- **Calibration** may influence expert ranking and pack construction.
- **Validation** may choose residency percentage and acceptance thresholds.
- **Held-out** must only be used for final association and quality checks.

Every prompt has a stable ID. Domain descriptions state what is measured and what is deliberately excluded. `contrast_domains` identify nearby abilities that should not be collapsed into the same label.

Validate or compile the library with:

```powershell
python scripts/mri_domain_library.py validate
python scripts/mri_domain_library.py compile --split calibration --out results/mri_library/calibration.json
python scripts/mri_domain_library.py catalog --out docs/tables/mri_domain_catalog.csv
```

The earlier `config/mri_diagnostic_payloads.json` remains frozen as the six-domain reproduction suite used by the first cross-model MRI result.
