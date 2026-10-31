# Adaptive v2: measured validation comparison

On the expanded amplitude-constrained validation cohort, K2/K3 adaptive v2 achieved **5/19 targeted successes (26.32%)**, compared with **4/19 (21.05%)** for PGD and counterfactual PGD. This is one additional targeted success, or **+5.26 percentage points**. Under the spectral constraint all four methods achieved **2/19 (10.53%)**.

These are automated nonclinical, validation-only measurements. They do not establish universal superiority, statistical significance, or a clinical incorrect-release improvement.

## Controlled comparison

- Dataset: prepared PTB-XL validation fold 9 only; no held-out evaluation was run.
- Target: T1, MI to NORM / released false-normal claim; shipped V1 verifier.
- Patients: all 19 clean-correct patients among the original 30 candidate patients; one ECG per patient.
- All nine attacks use the exact same patient/record/seed/budget/defense keys.
- Seed 42; one complete-system query per case. This severely limits decision-only methods and does not predict their ordering at larger query budgets.
- Raw-mV bounds: maximum perturbation 0.02 mV; spectral arm also limits >40 Hz energy ratio to 0.05 and adjacent perturbation slew to 0.01 mV.
- Gradient attacks: 40 steps and the same query budget. V2 changes objective, momentum (0.75), step adaptation and auxiliary weighting; this is a recipe comparison, not an isolated ablation of each component.
- Selected constraint validity: 100% for every arm. E1 has 342 complete submissions; the sealed log has 714 events and 372 waveform artifacts including 30 E0 cases.

## Complete outcomes

All cells have 19 patients. Released-target-claim successes equal targeted-diagnosis successes in this run.

| Attack | Amplitude target hits | Amplitude ASR | Spectral target hits | Spectral ASR | Amplitude untargeted hits | Spectral untargeted hits |
|---|---:|---:|---:|---:|---:|---:|
| f0 | 0/19 | 0.00% | 0/19 | 0.00% | 0/19 | 0/19 |
| f1_pgd | 4/19 | 21.05% | 2/19 | 10.53% | 11/19 | 5/19 |
| f1_cf | 4/19 | 21.05% | 2/19 | 10.53% | 8/19 | 5/19 |
| f2_nes | 0/19 | 0.00% | 0/19 | 0.00% | 0/19 | 0/19 |
| f2_simba | 0/19 | 0.00% | 0/19 | 0.00% | 0/19 | 0/19 |
| f2_square | 0/19 | 0.00% | 0/19 | 0.00% | 0/19 | 0/19 |
| f3_k1 | 0/19 | 0.00% | 0/19 | 0.00% | 0/19 | 0/19 |
| f3_k2 | 5/19 | 26.32% | 2/19 | 10.53% | 10/19 | 6/19 |
| f3_k3 | 5/19 | 26.32% | 2/19 | 10.53% | 10/19 | 6/19 |

The target-specific improvement has an untargeted trade-off: amplitude-only PGD finds 11/19 cases with any wrong diagnosis, versus 10/19 for adaptive v2. Spectral untargeted success improves from 5/19 for PGD to 6/19 for v2. Targeted and untargeted success must not be substituted for one another.

## Uncertainty and selection history

The initial ten-patient pilot increased target probabilities but tied exact-target success at 20% (amplitude) and 10% (spectral). The larger comparison uses every clean-correct patient in the same original candidate pool, without selecting individual successful patients. It remains a validation model-selection result; it is not independent confirmation.

- K3 v2 minus PGD, s_inf: +5.26 pp, nominal patient-bootstrap 95% interval [0.00, 15.79] pp.
- K3 v2 minus PGD, s_spec: +0.00 pp, nominal patient-bootstrap 95% interval [0.00, 0.00] pp.

Intervals use 10,000 patient resamples and are pointwise, with no multiplicity correction. The interval includes zero; a larger, multi-seed study and a newly locked held-out evaluation are needed for a stronger conclusion. Independent clean-report/adult/preserved-finding evidence is unavailable, so the clinical primary endpoint remains **not evaluable**, rather than zero.

## Reproduce

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 care-ecg security-run \
  artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/validation.npz \
  results/security-adaptive-v2-expanded-replay \
  --security-config config/security_ptbxl_adaptive_v2_validation.yaml \
  --phase validation --budgets 1 --seeds 42 \
  --max-eligible-per-target 30
```

The configuration defaults to ten eligible patients, multiple budgets and three seeds; the explicit overrides above identify the measured 19-patient study. Use the uncapped default configuration for a new full-patient study. Existing locked historical outputs and the earlier PDF are preserved.

## Local evidence and integrity

- [Full JSON](../results/security-adaptive-v2-expanded-validation-20261028/security_results.json)
- [Excel workbook](../results/security-adaptive-v2-expanded-validation-20261028/security_results.xlsx)
- Results SHA-256: `1f37a5fc9846bec2625fbbfaf8cf8229e48ae4d2d94b6c2d1c1fff5003b45a30`
- Validation dataset SHA-256: `b15dc0b71d0400d0697d5b0cbdb4362aaf093ebeb3521d2ec1a21e0a4812755e`
- Victim weights SHA-256: `ae32c49fa341fede8c503cd055df6d39648582197fa95edd8fced533827b2fad`
- Code-tree identity: `e47567a120704f4b1c2fd5a4c42cef765c58b8647f4016169970acac33ad42cf`

All original and new tests pass: **265 tests and 103 subtests**. The new tests exercise overshoot recovery, opposing verifier proxies, real rejection feedback, deterministic norm/spectral membership, query limits, and independence from clinical review labels.
