"""Aggregate-only evidence for adaptive v2 and finite-action DPO studies.

Uses frozen local artifacts. No model training, outcome selection or downloading.
Preference records are read for verification but patient identifiers are not saved.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def _preference_margins(checkpoint, features, chosen, rejected):
    logits = features @ np.asarray(checkpoint["weight"]).T + np.asarray(checkpoint["bias"])
    rows = np.arange(len(features))
    return logits[rows, chosen] - logits[rows, rejected]


def extend(evidence, root, out, inputs):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(relative):
        path = root / relative
        inputs[relative] = digest(path)
        return json.loads(path.read_text())

    def table(name, columns, rows):
        def esc(value):
            return str(value).replace("_", r"\_").replace("%", r"\%")

        with (out / "tables" / (name + ".csv")).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            w.writerows(rows)
        lines = [
            r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
            r"\toprule",
            " & ".join(map(esc, columns)) + r" \\",
            r"\midrule",
        ]
        lines += [" & ".join(map(esc, row)) + r" \\" for row in rows]
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / "tables" / (name + ".tex")).write_text("\n".join(lines) + "\n")

    def save(name):
        for ext in ("pdf", "png"):
            plt.savefig(out / "figures" / (name + "." + ext), bbox_inches="tight", dpi=170)
        plt.close()

    def key(obs):
        return tuple(
            obs[k] for k in ("patient_id", "record_id", "target_id", "seed", "budget", "defense")
        )

    base = "results/security-adaptive-v2-expanded-validation-20261028/"
    data = read(base + "security_results.json")
    saved_comparison = read(base + "adaptive_v2_comparison_summary.json")
    assert saved_comparison["source_results_sha256"] == inputs[base + "security_results.json"]
    assert data["protocol"]["phase"] == "validation"
    assert data["protocol"]["seeds"] == [42] and data["protocol"]["budgets"] == [1]
    assert (
        data["event_log"]["verification"]["valid"] and data["event_log"]["verification"]["complete"]
    )
    arms = ["f0", "f1_pgd", "f1_cf", "f2_nes", "f2_simba", "f2_square", "f3_k1", "f3_k2", "f3_k3"]
    names = ["F0", "PGD", "F1-CF", "NES", "SimBA", "Square", "K1", "K2 v2", "K3 v2"]
    constraints = ["s_inf", "s_spec"]
    outcomes, rows = {}, []
    paired = []
    event_fields = [
        "targeted_diagnosis_success",
        "untargeted_diagnosis_success",
        "released_target_claim_success",
        "selected_constraint_satisfied",
    ]
    for arm in arms:
        outcomes[arm] = {}
        row = [arm]
        for constraint in constraints:
            observations = data["e1"][arm][constraint]["observations"]
            by_key = {key(o): o["metrics"]["automated_robustness"] for o in observations}
            baseline = {
                key(o): o["metrics"]["automated_robustness"]
                for o in data["e1"]["f1_pgd"][constraint]["observations"]
            }
            assert len(by_key) == len(observations) == 19 and set(by_key) == set(baseline)
            assert len({k[0] for k in by_key}) == 19
            assert all(o["queries_used"] == 1 for o in observations)
            assert all(v["clean_correct"] for v in by_key.values())
            counts = {field: sum(bool(v[field]) for v in by_key.values()) for field in event_fields}
            assert counts["selected_constraint_satisfied"] == 19
            assert counts["targeted_diagnosis_success"] == counts["released_target_claim_success"]
            for field in event_fields[:3]:
                assert counts[field] == saved_comparison["outcomes"][arm][constraint][field]
            outcomes[arm][constraint] = {
                "patients": 19,
                "counts": counts,
                "rates": {field: value / 19 for field, value in counts.items()},
            }
            row += [
                f"{counts[event_fields[0]]}/19",
                f"{100 * counts[event_fields[0]] / 19:.2f}",
                f"{counts[event_fields[1]]}/19",
            ]
            if arm in ("f3_k2", "f3_k3"):
                keys = sorted(by_key)
                delta = np.array(
                    [
                        float(by_key[k][event_fields[0]]) - float(baseline[k][event_fields[0]])
                        for k in keys
                    ]
                )
                rng = np.random.default_rng(42)
                resampled = delta[rng.integers(0, len(delta), (10_000, len(delta)))].mean(1)
                ci = np.quantile(resampled, [0.025, 0.975]).tolist()
                paired.append(
                    {
                        "arm": arm,
                        "constraint": constraint,
                        "delta": float(delta.mean()),
                        "ci": ci,
                        "resamples": 10000,
                        "seed": 42,
                    }
                )
        rows.append(row)
    table(
        "adaptive_v2",
        ["Attack", "Amp hits", "Amp %", "Amp any", "Spec hits", "Spec %", "Spec any"],
        rows,
    )
    table(
        "adaptive_v2_paired",
        ["Contrast", "Set", "Delta (pp)", "95% interval (pp)"],
        [
            [
                p["arm"] + " - PGD",
                p["constraint"],
                f"{100 * p['delta']:.2f}",
                f"[{100 * p['ci'][0]:.2f}, {100 * p['ci'][1]:.2f}]",
            ]
            for p in paired
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    y = np.arange(len(arms))
    for ax, constraint, title in zip(
        axes, constraints, ["Amplitude constraint", "Spectral constraint"]
    ):
        target = [100 * outcomes[a][constraint]["rates"][event_fields[0]] for a in arms]
        any_error = [100 * outcomes[a][constraint]["rates"][event_fields[1]] for a in arms]
        ax.plot(target, y - 0.12, "o", color="#206B7D", label="Target / released target")
        ax.plot(any_error, y + 0.12, "s", color="#B76A16", label="Any wrong diagnosis")
        ax.set(
            xlim=(-3, 67),
            xticks=[0, 20, 40, 60],
            yticks=y,
            yticklabels=names,
            xlabel="Success (%)",
            title=title,
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=True)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Adaptive v2: 19 validation patients, T1, seed 42, B=1", fontsize=12)
    fig.tight_layout()
    save("adaptive_v2")
    evidence["adaptive_v2"] = {
        "scope": "validation-only automated nonclinical",
        "outcomes": outcomes,
        "paired_target_differences": paired,
        "protocol": data["protocol"],
        "log_verification": data["event_log"]["verification"],
        "historical_code_identity": data["implementation"].get("code_identity"),
        "all_case_keys_match": True,
        "observations": 342,
        "queries": 342,
    }

    dpo_base = "artifacts/security-dpo-evolution-verified-20261028/"
    manifest = read(dpo_base + "manifest.json")
    assert manifest["complete"] and manifest["clinical_primary_endpoint"] == "not_evaluable"
    assert manifest["event_log"]["valid"] and manifest["event_log"]["complete"]
    fit_rows, validation_rows, checks = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharey=True)
    for r in manifest["rounds"]:
        rd = dpo_base + f"round-{r['round']:03d}/"
        for ax, role in zip(axes, ["attacker", "verifier"]):
            fit = r[role + "_fit"]
            parent = read(rd + role + "-reference.json")
            candidate = read(rd + role + "-candidate.json")
            path = root / (rd + role + "-preferences.jsonl")
            inputs[str(path.relative_to(root))] = digest(path)
            pairs = [json.loads(line) for line in path.read_text().splitlines()]
            # Reconstruct patient-balanced frozen-reference DPO from saved pairs.
            assert len(pairs) == fit["pairs"]
            assert len({p["patient_id"] for p in pairs}) == fit["patients"]
            actions = parent["actions"]
            features = np.asarray([p["features"] for p in pairs], dtype=np.float32).astype(float)
            chosen = np.asarray([actions.index(p["chosen"]) for p in pairs])
            rejected = np.asarray([actions.index(p["rejected"]) for p in pairs])
            reference_margin = _preference_margins(parent, features, chosen, rejected)
            hardness = 1 + 1 / (1 + np.exp(reference_margin))
            totals = Counter()
            for pair, h in zip(pairs, hardness):
                totals[pair["patient_id"]] += h
            weights = np.array([h / totals[p["patient_id"]] for p, h in zip(pairs, hardness)])
            weights /= weights.sum()
            candidate_margin = _preference_margins(candidate, features, chosen, rejected)
            loss = float(
                np.dot(
                    weights, np.logaddexp(0, -fit["beta"] * (candidate_margin - reference_margin))
                )
            )
            assert np.isclose(loss, fit["final_loss"], atol=1e-9)
            assert parent["policy_sha256"] == fit["reference_sha256"]
            assert candidate["policy_sha256"] == fit["candidate_sha256"]
            assert candidate["policy_sha256"] != parent["policy_sha256"]
            assert fit["reference_frozen"] and not r[role + "_promoted"]
            checks.append(
                {
                    "round": r["round"],
                    "role": role,
                    "recomputed_loss": loss,
                    "loss_matches": True,
                    "candidate_differs_from_reference": True,
                }
            )
            fit_rows.append(
                [
                    r["round"],
                    role,
                    fit["pairs"],
                    fit["patients"],
                    f"{fit['initial_loss']:.6f}",
                    f"{fit['final_loss']:.6f}",
                    f"{100 * fit['training_preference_accuracy']:.2f}",
                    "No",
                ]
            )
            ax.plot(
                range(fit["epochs"] + 1),
                [fit["initial_loss"]] + fit["loss_history"],
                label=f"Round {r['round']}",
                lw=1.8,
                linestyle="-" if r["round"] == 1 else "--",
            )
            ax.set(title=role.capitalize() + " controller", xlabel="DPO epoch", ylim=(0, 0.75))
            ax.grid(alpha=0.2)
        for arm in ["baseline", "attacker_candidate", "verifier_candidate"]:
            v = r[arm]
            validation_rows.append(
                [
                    r["round"],
                    arm.replace("_candidate", " cand."),
                    f"{100 * v['targeted_released_claim_asr']:.1f}",
                    f"{100 * v['false_release_rate']:.1f}",
                    f"{100 * v['correct_clean_coverage']:.1f}",
                    v["pipeline_calls"],
                ]
            )
    axes[0].set_ylabel("Patient-balanced training DPO loss")
    axes[1].legend()
    fig.suptitle("Training optimization succeeded; validation promoted neither role", fontsize=11)
    fig.tight_layout()
    save("dpo_training")
    table(
        "dpo_fit",
        ["Round", "Role", "Pairs", "Patients", "Initial", "Best", "Pref. %", "Promoted"],
        fit_rows,
    )
    table(
        "dpo_validation",
        ["Round", "Validation arm", "Target %", "False release %", "Coverage %", "Calls"],
        validation_rows,
    )
    for role in ["attacker", "verifier"]:
        incumbent = read(dpo_base + role + ".json")
        assert incumbent["policy_sha256"] == manifest[role + "_sha256"]
        assert (
            incumbent["policy_sha256"] == manifest["rounds"][0][role + "_fit"]["reference_sha256"]
        )

    run_base = "results/security-dpo-final-binding-check-20261028/"
    runtime = read(run_base + "security_results.json")
    lock = read(run_base + "security_protocol.lock.json")
    assert lock["validation_results_sha256"] == inputs[run_base + "security_results.json"]
    assert (
        lock["resolved_protocol"]["preference_policies"]
        == runtime["protocol"]["preference_policies"]
    )
    bindings = []
    for name, binding in lock["resolved_protocol"]["preference_policies"].items():
        path = root / binding["path"]
        assert digest(path) == binding["file_sha256"]
        checkpoint = json.loads(path.read_text())
        assert checkpoint["policy_sha256"] == binding["policy_sha256"]
        bindings.append(
            {
                "role": name,
                "file_sha256": binding["file_sha256"],
                "policy_sha256": binding["policy_sha256"],
                "verified": True,
            }
        )
    runtime_counts = {}
    for arm, by_constraint in runtime["e1"].items():
        runtime_counts[arm] = {}
        for constraint, cell in by_constraint.items():
            observations = cell["observations"]
            runtime_counts[arm][constraint] = {
                "observations": len(observations),
                **{
                    field: sum(
                        bool(o["metrics"]["automated_robustness"][field]) for o in observations
                    )
                    for field in event_fields
                },
            }
    evidence["dpo"] = {
        "complete": manifest["complete"],
        "config": manifest["config"],
        "cohorts": {
            k: manifest[k]
            for k in [
                "training_cases",
                "validation_cases",
                "training_patients",
                "validation_patients",
            ]
        },
        "rounds": manifest["rounds"],
        "loss_reconstruction_checks": checks,
        "historical_training_code_sha256": manifest["provenance"]["code"]["sha256"],
        "victim_model_tree_sha256": manifest["provenance"]["model"]["sha256"],
        "event_log": manifest["event_log"],
        "stopped_reason": manifest["stopped_reason"],
        "clinical_primary_endpoint": manifest["clinical_primary_endpoint"],
        "llm_finetuning": False,
        "runtime_counts": runtime_counts,
        "runtime_log": runtime["event_log"]["verification"],
        "runtime_policy_bindings": bindings,
        "exported_incumbents_are_initial_priors": True,
    }
    # Only aggregate metrics, hashes, feature contracts, and curves are exported.
    # No preference feature vectors or patient/record identifiers are exported.
    selected = [
        base + "security_results.json",
        base + "adaptive_v2_comparison_summary.json",
        dpo_base + "manifest.json",
        run_base + "security_results.json",
        run_base + "security_protocol.lock.json",
        dpo_base + "round-001/attacker-candidate.json",
        dpo_base + "round-001/verifier-candidate.json",
    ]
    ledger = [
        r"\begin{longtable}{p{0.55\linewidth}p{0.38\linewidth}}",
        r"\toprule",
        r"New experiment input & SHA-256 (two consecutive halves) \\",
        r"\midrule\endhead",
    ]
    for relative in selected:
        value = inputs[relative]
        ledger.append(
            r"{\scriptsize\path{"
            + relative
            + r"}} & {\scriptsize\ttfamily "
            + value[:32]
            + r"\newline "
            + value[32:]
            + r"} \\"
        )
    ledger += [r"\bottomrule", r"\end{longtable}"]
    (out / "tables" / "new_provenance.tex").write_text("\n".join(ledger) + "\n")
