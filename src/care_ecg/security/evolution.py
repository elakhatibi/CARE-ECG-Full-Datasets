"""Bounded attacker/verifier co-evolution with fresh rollouts and DPO updates.

Training uses reference diagnostic labels as an explicit nonclinical proxy. It
never treats the learned verifier's own verdict as correctness supervision.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .attacks import F3AdaptiveAttack
from .learned import (
    ATTACK_RECIPES,
    DPOAdaptiveAttack,
    attacker_features,
    feature_names,
    snapshot_features,
)
from .logging import AppendOnlySecurityLogger
from .preferences import PreferencePair, PreferencePolicy, fit_dpo


@dataclass(frozen=True)
class EvolutionConfig:
    rounds: int = 3
    patience: int = 2
    beta: float = 0.5
    dpo_epochs: int = 80
    learning_rate: float = 0.03
    max_pairs: int | None = None
    query_budget: int = 1
    collection_actions: int = 3
    steps: int = 40
    restarts: int = 1
    step_size_mv: float = 0.002
    confidence: float = 3.0
    seed: int = 42
    min_improvement: float = 0.0001
    min_clean_coverage: float = 0.9
    max_coverage_drop: float = 0.02

    def __post_init__(self):
        for name in (
            "rounds",
            "patience",
            "dpo_epochs",
            "query_budget",
            "collection_actions",
            "steps",
            "restarts",
        ):
            if (
                isinstance(getattr(self, name), bool)
                or not isinstance(getattr(self, name), int)
                or getattr(self, name) < 1
            ):
                raise ValueError(f"{name} must be positive")
        if self.max_pairs is not None and (
            isinstance(self.max_pairs, bool)
            or not isinstance(self.max_pairs, int)
            or self.max_pairs < 1
        ):
            raise ValueError("max_pairs must be positive or null")
        if self.collection_actions > len(ATTACK_RECIPES):
            raise ValueError("collection_actions exceeds the audited recipe catalog")
        for name in ("beta", "learning_rate", "min_improvement"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("step_size_mv", "confidence"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.min_clean_coverage <= 1 or not 0 <= self.max_coverage_drop <= 1:
            raise ValueError("Coverage constraints must be in [0,1]")


def validate_case_splits(train, validation):
    if not train or not validation:
        raise ValueError("Evolution needs nonempty training and validation cohorts")
    train_patients = {case.patient_id for case in train}
    validation_patients = {case.patient_id for case in validation}
    if train_patients & validation_patients:
        raise ValueError("Evolution train/validation patients overlap")
    if {case.record_id for case in train} & {case.record_id for case in validation}:
        raise ValueError("Evolution train/validation records overlap")
    for cases, allowed in [(train, set(range(1, 9))), (validation, {9})]:
        if any(case.metadata.get("fold") not in allowed for case in cases):
            raise ValueError("Evolution requires folds 1-8 for training and fold 9 for validation")


def _patient_mean(values):
    grouped = defaultdict(list)
    for patient, value in values:
        grouped[patient].append(float(value))
    return None if not grouped else float(np.mean([np.mean(v) for v in grouped.values()]))


def promotion_decision(role, baseline, candidate, config):
    """Compare to the same frozen opponent; never accept reject-everything gains."""
    if role == "attacker":
        field = "targeted_released_claim_asr"
        a, b = baseline[field], candidate[field]
        promote = (
            all(v is not None and np.isfinite(v) and 0 <= v <= 1 for v in (a, b))
            and b >= a + config.min_improvement
        )
        return bool(
            promote
        ), "targeted_release_asr_improved" if promote else "no_validated_attack_gain"
    if role != "verifier":
        raise ValueError("Unknown evolution role")
    a, b = baseline["false_release_rate"], candidate["false_release_rate"]
    coverage = candidate["correct_clean_coverage"]
    previous = baseline["correct_clean_coverage"]
    if any(
        value is None or not np.isfinite(value) or not 0 <= value <= 1
        for value in (a, b, coverage, previous)
    ):
        return False, "insufficient_validation_correctness_or_coverage_evidence"
    if coverage < config.min_clean_coverage or coverage < previous - config.max_coverage_drop:
        return False, "clean_coverage_guard_failed"
    promote = b <= a - config.min_improvement
    return bool(
        promote
    ), "false_release_reduced_at_preserved_coverage" if promote else "no_validated_verifier_gain"


class _ObservedVictim:
    def __init__(self, victim, logger, context):
        self.victim, self.logger, self.context = victim, logger, context
        self.calls = 0
        self.records = []

    def __getattr__(self, name):
        return getattr(self.victim, name)

    def evaluate(self, waveform, target, **kwargs):
        snapshot = self.victim.evaluate(waveform, target, **kwargs)
        self.calls += 1
        self.records.append((np.asarray(waveform).copy(), copy.deepcopy(snapshot)))
        self.logger.append(
            "evolution_pipeline_query",
            {
                **self.context,
                "call_index": self.calls,
                "snapshot": snapshot.to_dict(),
            },
            waveform_mv=waveform,
        )
        return snapshot


def _attack_kwargs(config, constraint, seed, verifier):
    return {
        "epsilon_mv": float(constraint.epsilon_mv),
        "norm": "l2" if str(constraint.kind) == "s2" else "linf",
        "step_size_mv": config.step_size_mv,
        "steps": config.steps,
        "restarts": config.restarts,
        "confidence": config.confidence,
        "seed": seed,
        "defense_weight": 1.0 if verifier is not None else 0.0,
    }


def _verifier_pairs(case, observed, labels):
    pairs = []
    for waveform, snapshot in observed:
        # Reference label is a teacher only; it is absent from both role features.
        correct = snapshot.diagnosis == case.target.clean_label
        chosen = "release" if correct else "reject"
        pairs.append(
            PreferencePair(
                snapshot_features(snapshot, waveform, labels),
                chosen,
                "reject" if correct else "release",
                case.patient_id,
                "dataset_diagnosis_proxy_not_all_claims_review",
            )
        )
    return pairs


def collect_preferences(cases, attacker, verifier, provider, constraint, config, logger, round_id):
    attack_pairs, verifier_pairs = [], []
    total_queries = 0
    for case in cases:
        context = {
            "round": round_id,
            "phase": "training",
            "patient_id": case.patient_id,
            "record_id": case.record_id,
            "target_id": case.target.target_id,
            "attacker_sha256": attacker.fingerprint(),
            "verifier_sha256": verifier.fingerprint(),
        }
        victim = _ObservedVictim(provider(case=case, verifier_policy=verifier), logger, context)
        clean = victim.evaluate(case.waveform_mv, case.target.target_label)
        verifier_pairs.extend(_verifier_pairs(case, victim.records, attacker.label_order))
        if clean.diagnosis != case.target.clean_label:
            total_queries += victim.calls
            continue
        features = attacker_features(
            clean,
            case.waveform_mv,
            attacker.label_order,
            case.target.target_label,
            config.query_budget,
            constraint,
        )
        # Include exploitation plus round-dependent exploration. Each candidate is
        # a separate budgeted episode; the ledger exposes total collection compute.
        with np.errstate(all="raise"):
            order = sorted(
                attacker.actions,
                key=lambda a: (
                    a != attacker.choose(features),
                    (attacker.actions.index(a) - round_id) % len(attacker.actions),
                ),
            )[: config.collection_actions]
        outcomes = []
        for action in order:
            start = len(victim.records)
            attack = F3AdaptiveAttack(
                **{
                    **_attack_kwargs(config, constraint, config.seed + round_id, verifier),
                    **ATTACK_RECIPES[action],
                }
            )
            before = victim.calls
            result = attack.run(
                victim,
                case.waveform_mv,
                case.target.target_label,
                query_budget=config.query_budget,
                constraint=constraint,
                clean_snapshot=clean,
            )
            if (
                victim.calls - before != result.queries_used
                or result.queries_used > config.query_budget
            ):
                raise RuntimeError("Evolution attack violated complete-system query accounting")
            if any(not constraint.contains(case.waveform_mv, a.waveform) for a in result.attempts):
                raise RuntimeError("Evolution received an out-of-constraint attack")
            verifier_pairs.extend(
                _verifier_pairs(case, victim.records[start:], attacker.label_order)
            )
            event = any(a.released and a.target_claim_present for a in result.attempts)
            probability = max((a.target_probability for a in result.attempts), default=0.0)
            outcomes.append((action, 2.0 * event + 0.1 * probability))
        for left, right in itertools.combinations(outcomes, 2):
            if abs(left[1] - right[1]) <= 1e-5:
                continue
            winner, loser = (left, right) if left[1] > right[1] else (right, left)
            attack_pairs.append(
                PreferencePair(
                    features,
                    winner[0],
                    loser[0],
                    case.patient_id,
                    "released_target_then_probability_nonclinical",
                )
            )
        total_queries += victim.calls
    return attack_pairs, verifier_pairs, total_queries


def evaluate_policies(
    cases, attacker, verifier, provider, constraint, config, logger, round_id, arm
):
    targeted, incorrect_release, clean_coverage = [], [], []
    calls = attack_queries = 0
    for case in cases:
        victim = _ObservedVictim(
            provider(case=case, verifier_policy=verifier),
            logger,
            {
                "round": round_id,
                "phase": "validation",
                "arm": arm,
                "patient_id": case.patient_id,
                "record_id": case.record_id,
                "attacker_sha256": attacker.fingerprint(),
                "verifier_sha256": verifier.fingerprint(),
            },
        )
        clean = victim.evaluate(case.waveform_mv, case.target.target_label)
        clean_correct = clean.diagnosis == case.target.clean_label
        if clean_correct:
            clean_coverage.append((case.patient_id, clean.released))
            attack = DPOAdaptiveAttack(
                attacker, **_attack_kwargs(config, constraint, config.seed, verifier)
            )
            result = attack.run(
                victim,
                case.waveform_mv,
                case.target.target_label,
                query_budget=config.query_budget,
                constraint=constraint,
                clean_snapshot=clean,
            )
            if victim.calls - 1 != result.queries_used or result.queries_used > config.query_budget:
                raise RuntimeError("Validation attack query accounting failed")
            if any(not constraint.contains(case.waveform_mv, a.waveform) for a in result.attempts):
                raise RuntimeError("Validation attack violated waveform constraints")
            targeted.append(
                (
                    case.patient_id,
                    any(a.released and a.target_claim_present for a in result.attempts),
                )
            )
            attack_queries += result.queries_used
        # All actually queried wrong diagnoses contribute, not only the selected
        # final candidate. Correctness supervision is independent of release state.
        for _, snapshot in victim.records:
            if snapshot.diagnosis != case.target.clean_label:
                incorrect_release.append((case.patient_id, snapshot.released))
        calls += victim.calls
    return {
        "targeted_released_claim_asr": _patient_mean(targeted),
        "false_release_rate": _patient_mean(incorrect_release),
        "correct_clean_coverage": _patient_mean(clean_coverage),
        "pipeline_calls": calls,
        "attack_queries": attack_queries,
        "clean_correct_cases": len(clean_coverage),
        "incorrect_snapshots": len(incorrect_release),
        "patients": len({c.patient_id for c in cases}),
        "cases": len(cases),
        "statistical_unit": "patient; average repeated observations within patient",
        "endpoint": "automated_nonclinical_diagnosis_proxy",
    }


def evolve(
    train_cases: Sequence[Any],
    validation_cases: Sequence[Any],
    *,
    label_order,
    provider: Callable,
    constraint,
    output: str | Path,
    config: EvolutionConfig,
    provenance: dict | None = None,
):
    validate_case_splits(train_cases, validation_cases)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "care-ecg-coevolution.v1",
        "complete": False,
        "config": asdict(config),
        "provenance": provenance or {},
        "rounds": [],
        "training_cases": len(train_cases),
        "validation_cases": len(validation_cases),
        "training_patients": len({c.patient_id for c in train_cases}),
        "validation_patients": len({c.patient_id for c in validation_cases}),
        "preference_scope": "dataset_diagnosis_proxy; no clinical all-claims adjudication",
        "llm_finetuning": False,
        "clinical_primary_endpoint": "not_evaluable",
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger = AppendOnlySecurityLogger(root / "evolution_events.jsonl")
    attacker = PreferencePolicy(
        "attacker",
        feature_names(label_order, "attacker"),
        tuple(ATTACK_RECIPES),
        label_order,
        preferred_action="adaptive_v2",
    )
    verifier = PreferencePolicy(
        "verifier",
        feature_names(label_order, "verifier"),
        ("release", "reject"),
        label_order,
        preferred_action="release",
    )
    unchanged = 0
    try:
        for round_id in range(1, config.rounds + 1):
            directory = root / f"round-{round_id:03d}"
            directory.mkdir()
            attacker.save(directory / "attacker-reference.json")
            verifier.save(directory / "verifier-reference.json")
            attack_pairs, verifier_pairs, collection_calls = collect_preferences(
                train_cases, attacker, verifier, provider, constraint, config, logger, round_id
            )
            for role, pairs in [("attacker", attack_pairs), ("verifier", verifier_pairs)]:
                (directory / f"{role}-preferences.jsonl").write_text(
                    "".join(json.dumps(asdict(pair)) + "\n" for pair in pairs)
                )
            kwargs = {
                "beta": config.beta,
                "epochs": config.dpo_epochs,
                "learning_rate": config.learning_rate,
                "max_pairs": config.max_pairs,
            }
            candidate_a, fit_a = fit_dpo(attacker, attack_pairs, **kwargs)
            candidate_v, fit_v = fit_dpo(verifier, verifier_pairs, **kwargs)
            candidate_a.save(directory / "attacker-candidate.json", metadata=fit_a)
            candidate_v.save(directory / "verifier-candidate.json", metadata=fit_v)
            baseline = evaluate_policies(
                validation_cases,
                attacker,
                verifier,
                provider,
                constraint,
                config,
                logger,
                round_id,
                "incumbents",
            )
            # Both comparisons use the same frozen opponent, fresh re-attacks,
            # identical seeds, patient cohort, query budget and constraint.
            attack_eval = evaluate_policies(
                validation_cases,
                candidate_a,
                verifier,
                provider,
                constraint,
                config,
                logger,
                round_id,
                "attacker_candidate",
            )
            verifier_eval = evaluate_policies(
                validation_cases,
                attacker,
                candidate_v,
                provider,
                constraint,
                config,
                logger,
                round_id,
                "verifier_candidate",
            )
            promote_a, reason_a = promotion_decision("attacker", baseline, attack_eval, config)
            promote_v, reason_v = promotion_decision("verifier", baseline, verifier_eval, config)
            joint = (
                evaluate_policies(
                    validation_cases,
                    candidate_a,
                    candidate_v,
                    provider,
                    constraint,
                    config,
                    logger,
                    round_id,
                    "proposed_joint",
                )
                if promote_a and promote_v
                else attack_eval
                if promote_a
                else verifier_eval
                if promote_v
                else baseline
            )
            record = {
                "round": round_id,
                "attacker_fit": fit_a,
                "verifier_fit": fit_v,
                "baseline": baseline,
                "attacker_candidate": attack_eval,
                "verifier_candidate": verifier_eval,
                "joint_after_selection": joint,
                "comparison": "each candidate versus the same frozen opposing incumbent",
                "attacker_promoted": promote_a,
                "verifier_promoted": promote_v,
                "attacker_reason": reason_a,
                "verifier_reason": reason_v,
                "collection_pipeline_calls": collection_calls,
            }
            if promote_a:
                attacker = candidate_a
            if promote_v:
                verifier = candidate_v
            record["next_attacker_sha256"] = attacker.fingerprint()
            record["next_verifier_sha256"] = verifier.fingerprint()
            logger.append("evolution_round", record)
            (directory / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
            manifest["rounds"].append(record)
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            unchanged = 0 if promote_a or promote_v else unchanged + 1
            if unchanged >= config.patience:
                break
        final_metadata = {
            "selection": "validated_incumbent",
            "victim_model_tree_sha256": (provenance or {}).get("model", {}).get("sha256"),
        }
        logger.seal()
        verification = logger.verify(require_manifest=True).to_dict()
        if not verification["valid"] or verification["complete"] is not True:
            raise RuntimeError("Evolution log integrity verification failed")
        attacker.save(root / "attacker.json", metadata=final_metadata)
        verifier.save(root / "verifier.json", metadata=final_metadata)
        manifest.update(
            {
                "complete": True,
                "attacker_sha256": attacker.fingerprint(),
                "verifier_sha256": verifier.fingerprint(),
                "event_log": verification,
                "stopped_reason": "patience" if unchanged >= config.patience else "round_limit",
            }
        )
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
