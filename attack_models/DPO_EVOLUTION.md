# DPO and bounded attacker/verifier co-evolution

The pipeline now supports **two trainable finite-action policies**:

- An attacker policy chooses a bounded attack recipe before an episode.
- A verifier policy learns `release` versus `reject`, adding a veto to an existing
  verifier. It cannot override that verifier's rejection or missing evidence.

These are contextual categorical policies trained with the actual
[Direct Preference Optimization objective](https://arxiv.org/abs/2305.18290).
They do not fine-tune an LLM or the ECG classifier. The generation backend remains
independent of these controllers.

The [development validation report](DPO_EVOLUTION_VALIDATION.md) records two real
DPO training rounds, their loss reductions and non-promotion decisions, and the
audited runtime/validation-lock checks for both trained candidates.

## What evolves

Each generation starts from the last validated policy checkpoints. Fresh training
rollouts compare the audited `adaptive_v2`, `v2_no_momentum`, and `legacy` recipes
against the current verifier. Preferences train an attacker that chooses among
those recipes and a verifier that rejects incorrect model diagnoses. References
remain frozen throughout each DPO update.

Candidates are compared with their incumbents on separate validation patients.
Each comparison holds the opposing policy fixed, repeats seeds and budgets, and
re-attacks the proposed verifier through its differentiable policy surrogate.
The joint selected pair is also evaluated when both policies are promoted. The
next generation uses the selected checkpoints, not unconditionally the latest
training output. This is bounded recursive policy improvement; it does not rewrite
Python code or add unconstrained attack types.

Defaults are three rounds, a two-round no-improvement stopping rule, and an
explicit query budget for every episode. Existing model weights, historical
locks, evaluation metrics, and independent clinical evidence requirements remain
separate from the policy-learning process.

## DPO objective and supervision

For context `x`, preferred action `a+`, rejected action `a-`, and frozen reference
`pi_ref`, the per-pair loss is:

```text
-log sigmoid(beta * (
  log pi_theta(a+ | x) - log pi_theta(a- | x)
  - log pi_ref(a+ | x) + log pi_ref(a- | x)
))
```

The implementation optimizes log probabilities directly. It does not regress a
reward score and relabel that as DPO. It restores the best training-loss weights,
deduplicates identical pairs, balances the total contribution of each patient,
and prioritizes difficult preferences within each patient using the parent policy.
An optional pair cap is explicit; the default retains all available pairs.

Attacker preferences rank released target claims first, with target probability
as a secondary training signal. Every collection recipe runs as its own budgeted
episode, and the log exposes the total collection cost. The deployed attacker
chooses one recipe before spending its episode budget; recipe selection adds no
extra pipeline queries.

Verifier preferences come from a dataset-label correctness proxy, independent of
the learned verifier's current verdict. A correct diagnosis prefers release; an
incorrect diagnosis prefers rejection. Expected labels, patient IDs, fold numbers,
and attack-success labels are excluded from policy input features. Features use
diagnostic probabilities, waveform summaries, and observed evidence-risk fields.
The attack controller also receives the requested target and threat-budget context.

**The automatic teacher is a diagnosis-only, nonclinical proxy.** It does not
adjudicate every report claim or establish preserved clinical findings. Human or
independently adjudicated preferences can be supplied through `PreferencePair`
and `fit_dpo`; the built-in co-evolution runner labels its results as automated
nonclinical evidence and does not promote them to the clinical primary endpoint.

## Promotion and rollback

The attacker is promoted only if its patient-averaged released-target-claim ASR
improves against the same frozen verifier. The verifier is promoted only if its
incorrect-release rate decreases against the same frozen attacker while preserving
correct-clean coverage. Defaults require at least 90% correct-clean coverage and
allow at most a two-percentage-point drop from the incumbent. Rejecting every
report fails these checks.

No evidence of improvement means rollback to the incumbent. Insufficient clean
or incorrect-report validation examples also prevents verifier promotion.
Repeated use of fold 9 is model selection, not an independent test estimate.
Because attacker and verifier have opposing objectives, improvement against a
frozen opponent does not imply that both sides' final joint metrics improve
monotonically.

## Train both policies

The default configuration uses all matching patients and their records for the
configured targets. Training data must contain folds 1-8; validation must contain
fold 9 with disjoint patient and record IDs. Fold 10 is refused by this command.

```bash
care-ecg security-evolve \
  artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/train.npz \
  data/ptbxl-prepared/validation.npz \
  artifacts/security-dpo-evolution \
  --evolution-config config/security_evolution.yaml
```

Full-cohort collection is expensive. For an explicitly small integration run:

```bash
care-ecg security-evolve \
  artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/train.npz \
  data/ptbxl-prepared/validation.npz \
  artifacts/security-dpo-evolution-smoke \
  --rounds 2 --steps 8 --dpo-epochs 40 --targets T1 \
  --max-train-patients-per-target 6 \
  --max-validation-patients-per-target 6
```

The CLI records these overrides, actual patient/case counts, folds, hashes,
execution device, and collection budgets. It writes a new directory rather than
overwriting an existing experiment.

## Artifacts and audit

```text
manifest.json                 # completion, source hashes, rounds, promotions
attacker.json                 # final validated incumbent (possibly the prior)
verifier.json                 # final validated incumbent (possibly the prior)
round-001/
  attacker-reference.json
  verifier-reference.json
  attacker-preferences.jsonl
  verifier-preferences.jsonl
  attacker-candidate.json
  verifier-candidate.json
  evaluation.json
evolution_events.jsonl
evolution_events.jsonl.manifest.json
evolution_events_waveforms/
```

A trained candidate and a promoted incumbent are different artifacts. Training
can complete successfully with zero promotions; inspect `manifest.json` before
claiming performance improvements. Failed runs retain `complete: false` and an
error. Only fully completed runs export a final manifest with a verified sealed
event log. JSON policy checkpoints have checked shapes, roles, labels, and SHA-256
digests and require no pickle deserialization.

## Evaluate the learned policies in the existing security runner

Copy an existing security configuration and add the two checkpoint paths:

```yaml
protocol:
  verifier_policy: artifacts/security-dpo-evolution/verifier.json

attacks:
  f3_k3:
    preference_policy: artifacts/security-dpo-evolution/attacker.json
    defense_weight: 1.0
```

Keep the existing protocol fields, dataset hashes, constraints, and budgets. The
example `config/security_ptbxl_dpo.yaml` is ready to use after training the output
directory shown above:

```bash
care-ecg security-run artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/validation.npz results/security-dpo-validation \
  --security-config config/security_ptbxl_dpo.yaml --phase validation
```

The learned gate applies to all selected attack arms. The F3-K3 arm uses the
learned recipe selector, while comparator algorithms retain their own search.
The learned gate's surrogate is an approximation of the structured-feature gate;
actual release decisions and the original complete-system query oracle determine
outcomes. Learned-policy files are loaded once and their content hashes are bound
into the resolved protocol and any new validation lock. A changed policy file
cannot silently pass a held-out replay of that lock.

The new controller wrapper is an experimental E1 integration. It does not satisfy
the existing E5 exact-built-in-attack attestation merely by being DPO-trained.
Clinical-primary evaluation still requires the independent correctness,
preservation, adult-eligibility, and defense-specific evidence of the original
security protocol. No full-cohort or held-out DPO improvement is implied by the
development smoke test.
