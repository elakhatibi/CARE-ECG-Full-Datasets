# Feedback-adaptive waveform search (v2)

`F3AdaptiveAttack(search_version="adaptive_v2")` adds a stronger search procedure
for K2/K3 access. It aims to improve measured target-diagnosis and released-claim
attack success. It does not change the victim, success oracle, clinical endpoint,
waveform constraints, comparator algorithms, or test labels.

The bundled full-patient `config/security_ptbxl.yaml` enables v2 for K2 and K3.
The Python constructor defaults to `legacy`, so existing callers can reproduce the
earlier optimizer explicitly. K1 remains a decision-feedback attack and rejects
the v2 option, which requires differentiable surrogate access.

The [measured validation comparison](ADAPTIVE_V2_VALIDATION.md) reports 5/19
targeted amplitude-constrained successes for v2 versus 4/19 for PGD/F1-CF, with
spectral ties and an untargeted trade-off. It includes all nine attack methods,
the smaller pilot, paired intervals, source hashes, and exact replay overrides.

## Search changes

1. Optimize target negative log likelihood together with the requested margin.
   Target gradients remain available after the clipped margin first reaches zero.
2. Start with a norm-native step of
   `max(step_size_mv, adaptive_step_factor * epsilon_mv)`. At checkpoints, halve
   the step and return to the best candidate if progress stalls or oscillates.
   This is inspired by [Auto-PGD](https://arxiv.org/abs/2003.01690), but is a custom
   composite-objective optimizer rather than the reference AutoAttack algorithm.
3. While acquiring the target diagnosis, scale retrieval/verifier/defense losses
   by `auxiliary_scale`. A poorly calibrated verifier proxy can otherwise push
   against the diagnostic target before it is reached.
4. After an actual rejected target prediction, increase the verifier/defense
   penalties. Alternate fresh starts with refinement of an actual queried
   incumbent, using pipeline feedback rather than restarting identically.
5. Select within a restart using its surrogate objective. Select across restarts
   using one consistent complete-system feedback loss, because objectives with
   different phase weights are not directly comparable.

The attack never uses independent clinical-preservation judgments to choose a
candidate or stop its search. Every actual pipeline submission counts against the
same query budget. The first released target claim stops the observable search;
clinical success is scored separately with the original independent-evidence rule.

## Configuration

```yaml
attacks:
  f3_k2:
    search_version: adaptive_v2
    momentum_decay: 0.75
    adaptive_step_factor: 2.0
    auxiliary_scale: 0.1
  f3_k3:
    search_version: adaptive_v2
    momentum_decay: 0.75
    adaptive_step_factor: 2.0
    auxiliary_scale: 0.1
```

Both versions honor the requested steps, restarts, and full-system query budget.
The v2 trace records phase weights, actual feedback, warm-start query index,
step reductions, best iteration, and forward/backward evaluation counts. It does
not obtain extra complete-system calls to choose the best surrogate iterate.
Gradient-step and query budgets are different resources and must both be reported
when comparing white-box and decision-only methods.

## Reproducible validation comparison

Use the new explicit engineering-validation configuration to compare all shipped
attack implementations without modifying earlier locked runs:

```bash
care-ecg security-run artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/validation.npz \
  results/security-adaptive-v2-validation \
  --security-config config/security_ptbxl_adaptive_v2_validation.yaml \
  --phase validation
```

This configuration declares T1, ten selected clean-correct patients from up to 30
candidates, two constraints, budgets 1 and 3, and three seeds. The default full
configuration remains uncapped and retains all eligible records per patient.
For a smaller run, explicitly pass `--seeds 42`; report that override with results.

Compare the automated target-diagnosis and released-claim rates and paired patient
intervals on the exact same cases, budget, and perturbation limit. An improvement
is an experimental outcome, not a guaranteed ordering of attack names. Do not
select settings on fold 10. A new held-out run requires a new validation lock
binding the changed code/configuration, and should be run only after selection.

The attacked victim is the waveform-callable internal CARE model. Cached,
record-keyed external ensemble probabilities remain outside this threat model.
Without independent clean-report and target-specific preservation adjudication,
the clinical incorrect-release endpoint remains not evaluable.
