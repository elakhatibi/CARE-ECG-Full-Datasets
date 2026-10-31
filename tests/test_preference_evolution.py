import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from test_security_attacks import ToyVictim

from care_ecg.cli import _security_attack_factory, build_parser
from care_ecg.security.constraints import SInfConstraint
from care_ecg.security.contracts import ClinicalTarget
from care_ecg.security.defenses import ShippedVerifier
from care_ecg.security.evolution import (
    EvolutionConfig,
    evolve,
    promotion_decision,
    validate_case_splits,
)
from care_ecg.security.experiments import CohortCase
from care_ecg.security.learned import (
    ATTACK_RECIPES,
    DPOAdaptiveAttack,
    DPOVerifier,
    feature_names,
)
from care_ecg.security.preferences import PreferencePair, PreferencePolicy, dpo_loss, fit_dpo


class GatedToyVictim(ToyVictim):
    def __init__(self, policy):
        super().__init__()
        self.gate = DPOVerifier(policy)

    def evaluate(self, signal, target_label, **kwargs):
        snapshot = super().evaluate(signal, target_label, **kwargs)
        decision = self.gate.decide(snapshot, waveform_mv=signal)
        snapshot.released = decision.released
        snapshot.defense_score = decision.score
        snapshot.metadata["defense_decision"] = decision.to_dict()
        return snapshot

    def surrogate(self, signal, target_label, **kwargs):
        snapshot = super().surrogate(signal, target_label, **kwargs)
        snapshot.defense_loss = self.gate.surrogate(
            torch.as_tensor(signal), snapshot.probabilities, snapshot.latent
        )
        return snapshot


class PreferenceTests(unittest.TestCase):
    def test_saved_json_preference_pairs_can_be_replayed(self):
        pair = PreferencePair([0.2, 0.3], "a", "b", "p", "reference")
        self.assertEqual(pair.features, (0.2, 0.3))
        self.assertEqual(len({pair, pair}), 1)

    def test_dpo_matches_log_ratio_formula_and_freezes_reference_gradients(self):
        logp = torch.log_softmax(torch.tensor([[1.2, -0.4]], requires_grad=True), dim=1)
        reference = torch.log_softmax(torch.tensor([[0.1, 0.9]], requires_grad=True), dim=1)
        reference.retain_grad()
        loss = dpo_loss(logp, reference, torch.tensor([0]), torch.tensor([1]), beta=0.5)
        expected = -math.log(1 / (1 + math.exp(-0.5 * (1.6 - (-0.8)))))
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        loss.sum().backward()
        self.assertIsNone(reference.grad)

    def test_fit_increases_chosen_relative_odds_without_changing_parent(self):
        parent = PreferencePolicy("attacker", ["x"], ["a", "b"], ["clean", "target"])
        original = copy.deepcopy(parent.payload())
        pairs = [PreferencePair((1.0,), "b", "a", "p", "independent_outcome")]
        trained, stats = fit_dpo(parent, pairs, epochs=25)
        self.assertEqual(parent.payload(), original)
        self.assertGreater(
            float(trained.log_probs([1.0])[1].detach()), float(parent.log_probs([1.0])[1].detach())
        )
        self.assertLess(stats["final_loss"], stats["initial_loss"])
        self.assertTrue(stats["reference_frozen"])

    def test_duplicate_pairs_do_not_increase_a_patients_training_weight(self):
        parent = PreferencePolicy("verifier", ["x"], ["release", "reject"], ["clean", "target"])
        a = PreferencePair((1.0,), "release", "reject", "p1", "reference")
        b = PreferencePair((1.0,), "reject", "release", "p2", "reference")
        left, _ = fit_dpo(parent, [a, b], epochs=5)
        right, _ = fit_dpo(parent, [a] * 7 + [b], epochs=5)
        np.testing.assert_allclose(left.weight.detach(), right.weight.detach(), atol=1e-6)

    def test_json_checkpoint_roundtrip_and_tamper_detection(self):
        policy = PreferencePolicy("verifier", ["x"], ["release", "reject"], ["a", "b"])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "policy.json"
            policy.save(path)
            self.assertEqual(PreferencePolicy.load(path).fingerprint(), policy.fingerprint())
            payload = json.loads(path.read_text())
            payload["weight"][0][0] = 4.0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "digest"):
                PreferencePolicy.load(path)

    def test_zero_signal_verifier_surrogate_has_finite_input_gradients(self):
        labels = ("clean", "target")
        policy = PreferencePolicy(
            "verifier", feature_names(labels, "verifier"), ("release", "reject"), labels
        )
        with torch.no_grad():
            policy.weight[0].fill_(0.1)
        raw = torch.zeros(1, 12, 64, requires_grad=True)
        gate = DPOVerifier(policy)
        loss = gate.surrogate(raw, torch.tensor([[0.7, 0.3]]), torch.zeros(1, 1))
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(loss, raw)[0]).all()))

    def test_learned_gate_cannot_override_existing_gate_rejection(self):
        labels = ("clean", "target")
        policy = PreferencePolicy(
            "verifier",
            feature_names(labels, "verifier"),
            ("release", "reject"),
            labels,
            preferred_action="release",
        )
        snapshot = ToyVictim().evaluate(np.zeros((12, 64), dtype=np.float32), "target")
        snapshot.unsupported_claim_score = 0.2
        gate = DPOVerifier(policy, ShippedVerifier())
        decision = gate.decide(snapshot, waveform_mv=np.zeros((12, 64)))
        self.assertFalse(decision.released)
        self.assertTrue(decision.reasons)
        self.assertFalse(gate.decide(snapshot).evidence_available)

    def test_attacker_policy_preserves_budget_and_has_no_reference_label_features(self):
        labels = ("clean", "target")
        policy = PreferencePolicy(
            "attacker", feature_names(labels, "attacker"), tuple(ATTACK_RECIPES), labels
        )
        attack = DPOAdaptiveAttack(policy, epsilon_mv=0.1, steps=4, restarts=3)
        victim = ToyVictim(claim_threshold=np.inf)
        result = attack.run(
            victim, np.zeros((12, 64)), "target", query_budget=1, constraint=SInfConstraint(0.1)
        )
        self.assertEqual(result.queries_used, 1)
        self.assertEqual(result.metadata["preference_controller"]["extra_selection_queries"], 0)
        self.assertFalse(any("expected" in key or "patient" in key for key in policy.feature_names))

    def test_coverage_guard_rejects_the_reject_everything_verifier(self):
        config = EvolutionConfig()
        baseline = {"false_release_rate": 1.0, "correct_clean_coverage": 1.0}
        candidate = {"false_release_rate": 0.0, "correct_clean_coverage": 0.0}
        self.assertEqual(
            promotion_decision("verifier", baseline, candidate, config),
            (False, "clean_coverage_guard_failed"),
        )
        candidate["correct_clean_coverage"] = 1.0
        self.assertTrue(promotion_decision("verifier", baseline, candidate, config)[0])
        candidate["correct_clean_coverage"] = float("nan")
        self.assertFalse(promotion_decision("verifier", baseline, candidate, config)[0])

    def test_splits_reject_overlap_and_heldout_folds(self):
        target = ClinicalTarget("T", "clean", "target", "target finding", "fabrication")

        def case(patient, record, fold):
            return CohortCase(patient, record, np.zeros((12, 64)), target, {"fold": fold})

        with self.assertRaisesRegex(ValueError, "patients overlap"):
            validate_case_splits([case("p", "a", 1)], [case("p", "b", 9)])
        with self.assertRaisesRegex(ValueError, "folds"):
            validate_case_splits([case("a", "a", 1)], [case("b", "b", 10)])

    def test_bounded_recursive_training_uses_fresh_rollouts_and_exports_verified_artifacts(self):
        target = ClinicalTarget("T", "clean", "target", "target finding", "fabrication")
        train = [CohortCase("p1", "r1", np.zeros((12, 64)), target, {"fold": 1})]
        validation = [CohortCase("p2", "r2", np.zeros((12, 64)), target, {"fold": 9})]
        calls = []

        def provider(*, case, verifier_policy):
            calls.append((case.patient_id, verifier_policy.fingerprint()))
            return GatedToyVictim(verifier_policy)

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "evolution"
            result = evolve(
                train,
                validation,
                label_order=("clean", "target"),
                provider=provider,
                constraint=SInfConstraint(0.1),
                output=output,
                config=EvolutionConfig(rounds=2, patience=2, steps=4, dpo_epochs=12),
            )
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["rounds"]), 2)
            self.assertTrue(result["event_log"]["valid"])
            self.assertTrue(result["event_log"]["complete"])
            self.assertEqual(result["rounds"][0]["attacker_fit"]["status"], "trained")
            self.assertEqual(result["rounds"][0]["verifier_fit"]["status"], "trained")
            self.assertEqual(
                PreferencePolicy.load(output / "attacker.json").fingerprint(),
                result["attacker_sha256"],
            )
            for role in ["attacker", "verifier"]:
                pairs = [
                    json.loads(line)
                    for line in (output / "round-001" / f"{role}-preferences.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertTrue(pairs)
                self.assertEqual({p["patient_id"] for p in pairs}, {"p1"})
            self.assertGreater(sum(patient == "p2" for patient, _ in calls), 3)
            with self.assertRaises(FileExistsError):
                evolve(
                    train,
                    validation,
                    label_order=("clean", "target"),
                    provider=provider,
                    constraint=SInfConstraint(0.1),
                    output=output,
                    config=EvolutionConfig(),
                )

    def test_factory_and_parser_expose_trained_policy_workflow(self):
        args = build_parser().parse_args(
            ["security-evolve", "model", "train.npz", "validation.npz", "new-output"]
        )
        self.assertEqual(args.threads, 2)
        labels = ("clean", "target")
        policy = PreferencePolicy(
            "attacker", feature_names(labels, "attacker"), tuple(ATTACK_RECIPES), labels
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "policy.json"
            policy.save(path)
            _, factory = _security_attack_factory(
                "f3_k3", {"f3_k3": {"preference_policy": str(path)}}, SInfConstraint(0.1)
            )
            self.assertIsInstance(factory(seed=42, budget=1), DPOAdaptiveAttack)
