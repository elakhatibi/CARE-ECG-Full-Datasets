"""Behavioral checks for feedback-conditioned, budgeted adaptive waveform search."""

import json
import unittest

import numpy as np
import torch
from test_security_attacks import CASE, KnowledgeSeparatedVictim, RecordingAdmissibility, ToyVictim

from care_ecg.cli import _load_security_config, _security_attack_factory
from care_ecg.security.attacks import F1PGDAttack, F3AdaptiveAttack
from care_ecg.security.constraints import S2Constraint, SInfConstraint, SSpecConstraint
from care_ecg.security.contracts import KnowledgeLevel


class QuadraticVictim(ToyVictim):
    """A narrow reachable target region that fixed large PGD steps skip."""

    @staticmethod
    def _probabilities(signal):
        mean = signal.mean(dim=(-2, -1))
        score = 2.0 - ((mean - 0.04) / 0.015).square()
        return torch.softmax(torch.stack((torch.zeros_like(score), score), dim=1), dim=1), mean

    def evaluate(self, signal, target_label, **kwargs):
        snapshot = super().evaluate(signal, target_label, **kwargs)
        target = snapshot.probabilities["target"] > 0.5
        snapshot.diagnosis = "target" if target else "clean"
        snapshot.target_claim_present = target
        snapshot.report = "Assessment supports target." if target else "Assessment supports clean."
        return snapshot


class OpposingVerifierVictim(KnowledgeSeparatedVictim):
    def surrogate(self, signal, target_label, **kwargs):
        snapshot = super().surrogate(signal, target_label, **kwargs)
        snapshot.verifier_loss = 40.0 * snapshot.verifier_loss
        return snapshot


class RejectingTargetVictim(ToyVictim):
    def evaluate(self, signal, target_label, **kwargs):
        snapshot = super().evaluate(signal, target_label, **kwargs)
        snapshot.released = False
        return snapshot


class AdaptiveSearchTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.zeros((12, 64), dtype=np.float32)

    def test_reducing_stalled_steps_reaches_target_missed_by_fixed_step_pgd(self):
        common = {"epsilon_mv": 0.1, "step_size_mv": 0.1, "steps": 12, "random_start": False}
        constraint = SInfConstraint(0.1)
        baseline = F1PGDAttack(**common).run(
            QuadraticVictim(), self.clean, "target", query_budget=1, constraint=constraint
        )
        adaptive = F3AdaptiveAttack(
            search_version="adaptive_v2",
            knowledge_level="k2",
            retrieval_weight=0.0,
            verifier_weight=0.0,
            **common,
        ).run(QuadraticVictim(), self.clean, "target", query_budget=1, constraint=constraint)
        self.assertFalse(baseline.metadata["observable_attack_goal_achieved"])
        self.assertTrue(adaptive.metadata["observable_attack_goal_achieved"])
        self.assertGreater(adaptive.attempts[0].metadata["step_size_reductions"], 0)
        self.assertFalse(adaptive.success)  # No independent clinical preservation was supplied.
        self.assertEqual(adaptive.queries_used, baseline.queries_used)
        self.assertEqual(adaptive.metadata["surrogate_backward_evaluations"], common["steps"])
        self.assertTrue(constraint.contains(self.clean, adaptive.adversarial_signal))

    def test_acquisition_limits_a_proxy_that_opposes_target_diagnosis(self):
        common = {"epsilon_mv": 0.1, "step_size_mv": 0.02, "steps": 8, "random_start": False}
        legacy = F3AdaptiveAttack(**common).run(
            OpposingVerifierVictim(), self.clean, "target", query_budget=1
        )
        adaptive = F3AdaptiveAttack(search_version="adaptive_v2", **common).run(
            OpposingVerifierVictim(), self.clean, "target", query_budget=1
        )
        self.assertFalse(legacy.metadata["observable_attack_goal_achieved"])
        self.assertTrue(adaptive.metadata["observable_attack_goal_achieved"])

    def test_rejected_target_drives_repair_and_warm_start_on_actual_feedback(self):
        result = F3AdaptiveAttack(
            search_version="adaptive_v2",
            epsilon_mv=0.1,
            step_size_mv=0.02,
            steps=6,
            restarts=5,
            random_start=False,
        ).run(RejectingTargetVictim(), self.clean, "target", query_budget=2)
        self.assertEqual(result.queries_used, 2)
        self.assertEqual(len(result.attempts), 2)
        first, second = result.metadata["feedback_trace"]
        self.assertEqual(first["feedback_phase"], "acquire_target")
        self.assertEqual(second["feedback_phase"], "repair_release")
        self.assertEqual(second["feedback_diagnosis"], "target")
        self.assertFalse(second["feedback_released"])
        self.assertEqual(second["warm_start_query_index"], 1)
        self.assertGreater(
            second["effective_weights"]["verifier"], first["effective_weights"]["verifier"]
        )
        self.assertEqual(result.metadata["surrogate_backward_evaluations"], 12)
        self.assertEqual(result.metadata["surrogate_forward_evaluations"], 27)
        self.assertFalse(result.success)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_deterministic_projection_with_actual_norm_and_spectral_sets(self):
        for constraint, norm in [
            (SInfConstraint(0.1), "linf"),
            (S2Constraint(0.5, per_lead=True), "l2"),
            (SSpecConstraint(0.1, 100.0, max_slew_mv=0.01), "linf"),
        ]:
            with self.subTest(constraint=constraint.kind):
                args = {
                    "search_version": "adaptive_v2",
                    "norm": norm,
                    "epsilon_mv": constraint.epsilon_mv,
                    "step_size_mv": 0.02,
                    "steps": 8,
                    "restarts": 2,
                    "momentum_decay": 0.75,
                    "seed": 17,
                }
                left = F3AdaptiveAttack(**args).run(
                    ToyVictim(claim_threshold=np.inf),
                    self.clean,
                    "target",
                    query_budget=2,
                    constraint=constraint,
                )
                right = F3AdaptiveAttack(**args).run(
                    ToyVictim(claim_threshold=np.inf),
                    self.clean,
                    "target",
                    query_budget=2,
                    constraint=constraint,
                )
                self.assertEqual(left.queries_used, 2)
                for a, b in zip(left.attempts, right.attempts):
                    np.testing.assert_array_equal(a.waveform, b.waveform)
                    self.assertTrue(constraint.contains(self.clean, a.waveform))

    def test_search_does_not_depend_on_primary_clinical_review_outcome(self):
        args = {
            "search_version": "adaptive_v2",
            "epsilon_mv": 0.1,
            "step_size_mv": 0.02,
            "steps": 8,
            "restarts": 3,
            "seed": 42,
        }
        results = [
            F3AdaptiveAttack(**args).run(
                ToyVictim(),
                self.clean,
                "target",
                query_budget=3,
                admissibility_checker=RecordingAdmissibility(preserved),
                case=CASE,
            )
            for preserved in [True, False]
        ]
        self.assertEqual(results[0].queries_used, results[1].queries_used)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        for a, b in zip(results[0].attempts, results[1].attempts):
            np.testing.assert_array_equal(a.waveform, b.waveform)

    def test_invalid_search_parameters_and_k1_access_are_rejected(self):
        for kwargs in [
            {"search_version": "unknown"},
            {"adaptive_step_factor": 0},
            {"adaptive_step_factor": np.inf},
            {"auxiliary_scale": 0},
            {"auxiliary_scale": 1.1},
            {"auxiliary_scale": np.nan},
            {"search_version": "adaptive_v2", "knowledge_level": KnowledgeLevel.K1},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                F3AdaptiveAttack(**kwargs)

    def test_cli_enables_v2_for_adaptive_arms_without_altering_comparator_optimizer(self):
        config = _load_security_config("config/security_ptbxl.yaml")
        for name in ["f3_k2", "f3_k3"]:
            _, factory = _security_attack_factory(name, config["attacks"], SInfConstraint(0.02))
            attack = factory(seed=42, budget=1)
            self.assertEqual(attack.search_version, "adaptive_v2")
            self.assertEqual(attack.momentum_decay, 0.75)
        _, factory = _security_attack_factory("f1_pgd", config["attacks"], SInfConstraint(0.02))
        self.assertEqual(type(factory(seed=42, budget=1)), F1PGDAttack)
        self.assertEqual(factory(seed=42, budget=1).momentum_decay, 0.0)
