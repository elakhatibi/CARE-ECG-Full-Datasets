"""Deploy finite-action DPO controllers on real attack/verifier interfaces."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .attacks import F3AdaptiveAttack
from .contracts import AttackFamily, ReleaseDecision, VictimSnapshot
from .defenses import ShippedVerifier
from .preferences import PreferencePolicy

# Recipes cannot override epsilon, norm, query budget, steps, or restarts.
ATTACK_RECIPES = {
    "adaptive_v2": {"search_version": "adaptive_v2", "momentum_decay": 0.75},
    "v2_no_momentum": {"search_version": "adaptive_v2", "momentum_decay": 0.0},
    "legacy": {"search_version": "legacy", "momentum_decay": 0.0},
}
_SUMMARY_NAMES = (
    "entropy",
    "probability_margin",
    "waveform_mean",
    "waveform_rms",
    "waveform_std",
    "waveform_peak",
    "waveform_difference_rms",
    "hallucination",
    "unsupported",
    "fallback",
    "abstained",
)


def feature_names(label_order, role):
    names = tuple(f"p:{label}" for label in label_order) + _SUMMARY_NAMES
    if role == "attacker":
        names += tuple(f"target:{label}" for label in label_order)
        names += ("query_budget", "epsilon", "s_inf", "s2", "s_spec", "s_phys")
    return names


def _features(raw: Tensor, probabilities: Tensor, snapshot: VictimSnapshot | None) -> Tensor:
    if raw.ndim == 2:
        raw = raw.unsqueeze(0)
    if probabilities.ndim == 1:
        probabilities = probabilities.unsqueeze(0)
    if raw.ndim != 3 or probabilities.ndim != 2 or len(raw) != len(probabilities):
        raise ValueError("Preference context requires aligned waveform/probability batches")
    if raw.shape[-1] < 2 or not bool(torch.isfinite(raw).all()):
        raise ValueError("Preference waveform must be finite with at least two samples")
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("Preference probabilities must be finite and nonnegative")
    if not bool(
        torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[:, 0]), atol=1e-5)
    ):
        raise ValueError("Preference probabilities must have unit mass")
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    top = probabilities.topk(2, dim=-1).values
    axes = (-2, -1)
    rms = torch.sqrt(raw.square().mean(axes) + 1e-12)
    std = torch.sqrt(raw.var(axes, unbiased=False) + 1e-12)
    difference = torch.sqrt(torch.diff(raw, dim=-1).square().mean(axes) + 1e-12)
    columns = [
        entropy / np.log(probabilities.shape[-1]),
        top[:, 0] - top[:, 1],
        torch.tanh(raw.mean(axes) / 0.25),
        torch.log1p(rms / 0.25),
        torch.log1p(std / 0.25),
        torch.log1p(raw.abs().amax(axes) / 0.25),
        torch.log1p(difference / 0.02),
    ]
    observed = (
        (0.0, 0.0, 0.0, 0.0)
        if snapshot is None
        else (
            snapshot.hallucination_score,
            snapshot.unsupported_claim_score,
            float(snapshot.used_fallback),
            float(snapshot.abstained),
        )
    )
    columns.extend(raw.new_full((len(raw),), float(value)) for value in observed)
    return torch.cat((probabilities, torch.stack(columns, dim=1)), dim=1)


def snapshot_features(snapshot, waveform, labels):
    if set(snapshot.probabilities) != set(labels):
        raise ValueError("Snapshot labels differ from preference policy labels")
    with torch.no_grad():
        values = _features(
            torch.as_tensor(waveform, dtype=torch.float32),
            torch.tensor([[snapshot.probabilities[k] for k in labels]]),
            snapshot,
        )
    return tuple(float(x) for x in values[0])


def attacker_features(snapshot, waveform, labels, target, budget, constraint):
    values = snapshot_features(snapshot, waveform, labels)
    if target not in labels or budget < 1:
        raise ValueError("Invalid attack target or query budget")
    kind = str(getattr(constraint, "kind", "s_inf"))
    return (
        values
        + tuple(float(label == target) for label in labels)
        + (
            float(np.log1p(budget) / np.log(101)),
            float(getattr(constraint, "epsilon_mv", 0.0) / 0.02),
            *(float(kind == value) for value in ("s_inf", "s2", "s_spec", "s_phys")),
        )
    )


class DPOVerifier:
    """Learned veto on top of an existing gate, never an override of its rejection."""

    def __init__(self, policy: PreferencePolicy, base=None, *, threshold=0.5):
        if policy.role != "verifier" or policy.actions != ("release", "reject"):
            raise ValueError("DPO verifier requires a release/reject verifier policy")
        if policy.feature_names != feature_names(policy.label_order, "verifier"):
            raise ValueError("Unsupported verifier preference feature contract")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("DPO verifier threshold must be in [0,1]")
        self.policy, self.base, self.threshold = policy, base or ShippedVerifier(), float(threshold)
        self.kind = self.base.kind
        self._last_snapshot = None

    def decide(self, snapshot, *, waveform_mv=None, **kwargs) -> ReleaseDecision:
        base = self.base.decide(snapshot, waveform_mv=waveform_mv, **kwargs)
        self._last_snapshot = snapshot
        if waveform_mv is None:
            return ReleaseDecision(
                self.kind,
                False,
                1.0,
                self.threshold,
                ("DPO verifier waveform context unavailable",),
                evidence_available=False,
            )
        features = snapshot_features(snapshot, waveform_mv, self.policy.label_order)
        with torch.no_grad():
            reject = float(self.policy.log_probs(features).exp()[1])
        released = base.released and reject <= self.threshold
        reasons = list(base.reasons)
        if reject > self.threshold:
            reasons.append("DPO preference gate rejected the report")
        return ReleaseDecision(
            self.kind,
            released,
            reject if base.released else max(reject, self.threshold + 1),
            self.threshold,
            tuple(reasons),
            evidence_available=base.evidence_available,
            metadata={
                "base": base.to_dict(),
                "policy_sha256": self.policy.fingerprint(),
                "policy_kind": "finite_action_dpo",
                "reject_probability": reject,
                "clinical_validation": False,
            },
        )

    def surrogate(self, raw, probabilities, latent):
        del latent
        values = _features(raw, probabilities, self._last_snapshot)
        # Report-level fields use the last actually observed report. Waveform and
        # probability features stay differentiable; the real gate remains the oracle.
        loss = -self.policy.log_probs(values)[:, 0].mean()
        return loss.to(raw.device)


class DPOAdaptiveAttack:
    """Choose one recipe before an episode; the chosen attack owns the full budget."""

    family = AttackFamily.F3

    def __init__(self, policy: PreferencePolicy, **attack_kwargs):
        if policy.role != "attacker" or not set(policy.actions) <= ATTACK_RECIPES.keys():
            raise ValueError("DPO attack policy contains unsupported recipes")
        if policy.feature_names != feature_names(policy.label_order, "attacker"):
            raise ValueError("Unsupported attacker preference feature contract")
        self.policy, self.attack_kwargs = policy, attack_kwargs

    def run(
        self,
        victim,
        clean_signal,
        target_label,
        *,
        query_budget,
        constraint=None,
        clean_snapshot=None,
        **kwargs,
    ):
        target = victim.label_names[victim.target_index(target_label)]
        if tuple(victim.label_names) != self.policy.label_order:
            raise ValueError("Attack policy label order differs from victim")
        snapshot = clean_snapshot or victim.evaluate(
            clean_signal, target, defense=kwargs.get("defense", "v1")
        )
        features = attacker_features(
            snapshot, clean_signal, self.policy.label_order, target, query_budget, constraint
        )
        action = self.policy.choose(features)
        attack = F3AdaptiveAttack(**{**self.attack_kwargs, **ATTACK_RECIPES[action]})
        result = attack.run(
            victim,
            clean_signal,
            target,
            query_budget=query_budget,
            constraint=constraint,
            clean_snapshot=snapshot,
            **kwargs,
        )
        result.metadata["preference_controller"] = {
            "policy_sha256": self.policy.fingerprint(),
            "action": action,
            "role": "attacker",
            "selection": "greedy_before_budgeted_episode",
            "extra_selection_queries": 0,
        }
        return result
