"""Finite-action DPO policies with frozen-reference, patient-balanced training.

These policies select attack recipes or release decisions; they do not fine-tune
an LLM. The objective is the DPO log-policy-ratio loss, not reward regression.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SCHEMA = "care-ecg-preference-policy.v1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PreferencePair:
    features: tuple[float, ...]
    chosen: str
    rejected: str
    patient_id: str
    source: str

    def __post_init__(self):
        if not self.patient_id.strip() or not self.source.strip():
            raise ValueError("Preference pairs need patient and judgment provenance")
        if self.chosen == self.rejected:
            raise ValueError("A preference pair must compare different actions")
        if not self.features or not np.isfinite(self.features).all():
            raise ValueError("Preference features must be finite and nonempty")
        object.__setattr__(self, "features", tuple(float(value) for value in self.features))


class PreferencePolicy(nn.Module):
    """Small contextual categorical policy, serialized as checked JSON arrays."""

    def __init__(
        self,
        role: str,
        feature_names: Sequence[str],
        actions: Sequence[str],
        label_order: Sequence[str],
        *,
        preferred_action: str | None = None,
    ):
        super().__init__()
        if role not in {"attacker", "verifier"}:
            raise ValueError("Policy role must be attacker or verifier")
        self.role = role
        self.feature_names = tuple(feature_names)
        self.actions = tuple(actions)
        self.label_order = tuple(label_order)
        for values in (self.feature_names, self.actions, self.label_order):
            if (
                not values
                or any(not str(x).strip() for x in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("Policy contracts must contain unique, nonempty names")
        if len(self.actions) < 2 or len(self.label_order) < 2:
            raise ValueError("Policies need at least two actions and diagnosis labels")
        self.weight = nn.Parameter(
            torch.zeros(len(self.actions), len(self.feature_names), dtype=torch.float64)
        )
        self.bias = nn.Parameter(torch.zeros(len(self.actions), dtype=torch.float64))
        if preferred_action is not None:
            with torch.no_grad():
                self.bias[self.actions.index(preferred_action)] = 4.0 if role == "verifier" else 1.0

    def log_probs(self, features: Tensor | Sequence[float]) -> Tensor:
        x = torch.as_tensor(features, dtype=self.weight.dtype, device=self.weight.device)
        if x.ndim not in {1, 2} or x.shape[-1] != len(self.feature_names):
            raise ValueError("Preference feature shape differs from the saved contract")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Preference inputs must be finite")
        return F.log_softmax(F.linear(x, self.weight, self.bias), dim=-1)

    def choose(self, features: Tensor | Sequence[float]) -> str:
        with torch.no_grad():
            values = self.log_probs(features)
            if values.ndim != 1:
                raise ValueError("Action selection requires one context")
            return self.actions[int(values.argmax())]

    def payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "role": self.role,
            "feature_names": list(self.feature_names),
            "actions": list(self.actions),
            "label_order": list(self.label_order),
            "weight": self.weight.detach().cpu().tolist(),
            "bias": self.bias.detach().cpu().tolist(),
        }

    def fingerprint(self) -> str:
        return canonical_hash(self.payload())

    def save(self, path: str | Path, *, metadata: dict | None = None):
        payload = self.payload()
        payload["policy_sha256"] = canonical_hash(payload)
        payload["metadata"] = metadata or {}
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> PreferencePolicy:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        metadata = payload.pop("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Preference checkpoint metadata must be an object")
        expected = payload.pop("policy_sha256")
        if expected != canonical_hash(payload) or payload.get("schema") != SCHEMA:
            raise ValueError("Preference checkpoint digest/schema is invalid")
        policy = cls(
            payload["role"], payload["feature_names"], payload["actions"], payload["label_order"]
        )
        weight, bias = np.asarray(payload["weight"]), np.asarray(payload["bias"])
        if weight.shape != tuple(policy.weight.shape) or bias.shape != tuple(policy.bias.shape):
            raise ValueError("Preference checkpoint parameter shapes are invalid")
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError("Preference checkpoint parameters must be finite")
        with torch.no_grad():
            policy.weight.copy_(torch.as_tensor(weight))
            policy.bias.copy_(torch.as_tensor(bias))
        policy.source_file_sha256 = hashlib.sha256(raw).hexdigest()
        policy.checkpoint_metadata = metadata
        return policy


def dpo_loss(
    policy_log_probs: Tensor,
    reference_log_probs: Tensor,
    chosen: Tensor,
    rejected: Tensor,
    *,
    beta: float,
) -> Tensor:
    """Per-pair -log sigmoid(beta * (policy log-ratio - reference log-ratio))."""
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    if policy_log_probs.ndim != 2 or policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("DPO policy/reference log probabilities must align")
    n = policy_log_probs.shape[0]
    if chosen.shape != (n,) or rejected.shape != (n,) or bool((chosen == rejected).any()):
        raise ValueError("DPO action pairs are invalid")
    indices = torch.arange(n, device=policy_log_probs.device)
    ratio = policy_log_probs[indices, chosen] - policy_log_probs[indices, rejected]
    reference = reference_log_probs.detach()
    ref_ratio = reference[indices, chosen] - reference[indices, rejected]
    return -F.logsigmoid(float(beta) * (ratio - ref_ratio))


def fit_dpo(
    parent: PreferencePolicy,
    pairs: Sequence[PreferencePair],
    *,
    beta: float = 0.5,
    epochs: int = 80,
    learning_rate: float = 0.03,
    max_pairs: int | None = None,
) -> tuple[PreferencePolicy, dict]:
    if (
        epochs < 1
        or (max_pairs is not None and max_pairs < 1)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Invalid DPO optimization limits")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be finite and positive")
    if not pairs:
        return copy.deepcopy(parent), {"status": "no_preferences", "pairs": 0}
    available_pairs = len(pairs)
    pairs = list(dict.fromkeys(pairs))
    if any(len(p.features) != len(parent.feature_names) for p in pairs):
        raise ValueError("Preference features do not match policy contract")
    # Mine errors/weak preferences using training pairs only. Validation judgments
    # are never passed to this function by the evolution runner.
    action_index = {name: i for i, name in enumerate(parent.actions)}
    x = torch.tensor([p.features for p in pairs], dtype=torch.float32)
    chosen = torch.tensor([action_index[p.chosen] for p in pairs], dtype=torch.long)
    rejected = torch.tensor([action_index[p.rejected] for p in pairs], dtype=torch.long)
    reference = copy.deepcopy(parent).cpu().eval()
    reference.requires_grad_(False)
    with torch.no_grad():
        ref_log = reference.log_probs(x)
        margins = ref_log.gather(1, chosen[:, None]) - ref_log.gather(1, rejected[:, None])
        keep = torch.argsort(margins[:, 0], stable=True)
        if max_pairs is not None:
            keep = keep[:max_pairs]
    x, chosen, rejected, ref_log = x[keep], chosen[keep], rejected[keep], ref_log[keep]
    selected = [pairs[int(i)] for i in keep]
    counts = Counter(p.patient_id for p in selected)
    hardness = (1.0 + torch.sigmoid(-margins[keep, 0])).double()
    patient_hardness = Counter()
    for pair, value in zip(selected, hardness):
        patient_hardness[pair.patient_id] += float(value)
    weights = torch.tensor(
        [float(value) / patient_hardness[p.patient_id] for p, value in zip(selected, hardness)],
        dtype=torch.float64,
    )
    weights /= weights.sum()
    policy = copy.deepcopy(parent).cpu().train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    losses = []
    with torch.no_grad():
        initial_loss = float(
            (dpo_loss(policy.log_probs(x), ref_log, chosen, rejected, beta=beta) * weights).sum()
        )
    best_loss = initial_loss
    best_state = copy.deepcopy(policy.state_dict())
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = (dpo_loss(policy.log_probs(x), ref_log, chosen, rejected, beta=beta) * weights).sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite DPO training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            updated_loss = float(
                (
                    dpo_loss(policy.log_probs(x), ref_log, chosen, rejected, beta=beta) * weights
                ).sum()
            )
        losses.append(updated_loss)
        if updated_loss < best_loss - 1e-12:
            best_loss, best_state = updated_loss, copy.deepcopy(policy.state_dict())
    policy.load_state_dict(best_state)
    with torch.no_grad():
        logp = policy.log_probs(x)
        accuracy = (
            (logp.gather(1, chosen[:, None]) > logp.gather(1, rejected[:, None])).float().mean()
        )
    return policy.eval(), {
        "status": "trained",
        "objective": "frozen_reference_dpo",
        "beta": beta,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "pairs": len(selected),
        "available_pairs": available_pairs,
        "duplicate_pairs_removed": available_pairs - len(pairs),
        "patients": len(counts),
        "initial_loss": initial_loss,
        "final_loss": best_loss,
        "loss_history": losses,
        "training_preference_accuracy": float(accuracy),
        "reference_sha256": parent.fingerprint(),
        "candidate_sha256": policy.fingerprint(),
        "reference_frozen": True,
        "hard_mining": "within_patient_parent_margin_weighting; optional explicit pair cap",
        "patient_weighting": "equal patient mass",
        "sources": sorted({p.source for p in selected}),
    }
