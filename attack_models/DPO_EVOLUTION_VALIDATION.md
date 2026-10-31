# DPO/co-evolution development validation — 28 October 2026

The new workflow completed two real PTB-XL development rounds. Both the attacker and verifier decision policies were trained with frozen-reference DPO. The small run demonstrated execution and loss reduction, but **neither candidate passed the validation-promotion gates**. Exported incumbent policies therefore retained their prior parameters.

## Cohorts and resources

- Training: folds 1-8, 7 ECG/target cases from 6 patients.
- Validation: fold 9, 9 ECG/target cases from 6 different patients.
- Explicit development limits: T1 only; six patients per split; all matching records retained.
- Two rounds, eight gradient steps per attack, one query per recipe episode, 40 DPO epochs, beta 0.5.
- Three recipe episodes per eligible training context are collection cost, not a one-query ensemble result.
- Reference labels supply diagnosis-only, nonclinical preferences. No independent all-claims clinical adjudication was available.

## Actual DPO training

| Round | Role | Preference pairs used | Training patients with preferences | Initial DPO loss | Best final DPO loss | Promoted |
|---|---|---:|---:|---:|---:|---|
| 1 | attacker | 14 | 4 | 0.693147 | 0.105057 | False |
| 1 | verifier | 22 | 6 | 0.693147 | 0.421950 | False |
| 2 | attacker | 14 | 4 | 0.693147 | 0.105057 | False |
| 2 | verifier | 22 | 6 | 0.693147 | 0.405486 | False |

Lower training loss is evidence that the DPO objective was optimized; it is not evidence of improved attack success or clinical verification. In both rounds the validation attack ASR stayed at zero and the diagnosis-proxy false-release rate stayed at one, with correct-clean coverage one. These observations failed the improvement requirements, so automatic promotion was refused.

## Runtime and lock integration

The genuinely trained round-1 candidate files were also loaded explicitly into the existing security runner for an experimental integration check. This did not replace the selected incumbents. That check exercised PGD and learned F3 on four eligible ECG cases, using the same query and waveform constraints.

- Training log: valid and complete, 136 records, 134 waveform artifacts.
- Runtime log: valid and complete, 25 records.
- Both trained checkpoint file hashes are included in the resolved protocol and its verified validation lock.
- Regression tests cover missing lock bindings, altered policy files, checkpoint tampering, query budgets, patient separation, frozen references, and reject-everything coverage collapse.
- 277 tests and 103 subtests passed; static checks passed.

## Local artifacts

- [Training manifest](../artifacts/security-dpo-evolution-verified-20261028/manifest.json)
- [Attacker candidate](../artifacts/security-dpo-evolution-verified-20261028/round-001/attacker-candidate.json)
- [Verifier candidate](../artifacts/security-dpo-evolution-verified-20261028/round-001/verifier-candidate.json)
- [Audited runtime result](../results/security-dpo-final-binding-check-20261028/security_results.json)
- [Validation lock](../results/security-dpo-final-binding-check-20261028/security_protocol.lock.json)

Training and runtime manifests record their respective source identities. These are development checks, not a full-cohort, multi-seed, held-out, or clinical improvement study. See [the workflow guide](DPO_EVOLUTION.md) for full matching-patient collection and versioned training commands.
