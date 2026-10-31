"""Rebuild aggregate tables and figures for the CARE-ECG technical manuscript.

Reads existing local artifacts only. Does not train, tune, download, or modify them.
Run from any working directory with the repository's scientific Python environment.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/care-ecg-report-matplotlib")

import csv
import hashlib
import json
import platform
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TAB = OUT / "tables"
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
SEED, RESAMPLES = 20261027, 10_000
INPUTS: dict[str, str] = {}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(relative: str):
    path = ROOT / relative
    INPUTS[relative] = digest(path)
    return json.loads(path.read_text())


def tex(value) -> str:
    return str(value).translate(str.maketrans({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}))


def table(name, columns, rows, alignment=None):
    with (TAB / (name + ".csv")).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)
    body = [
        r"\begin{tabular}{" + (alignment or "l" + "r" * (len(columns) - 1)) + "}",
        r"\toprule",
        " & ".join(tex(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    body += [" & ".join(tex(v) for v in row) + r" \\" for row in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    (TAB / (name + ".tex")).write_text("\n".join(body) + "\n")


def metric_arrays(cm):
    """Accept [..., true class, predicted class] counts; return four metrics."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=float),
        where=(precision + recall) > 0,
    )
    accuracy = tp.sum(axis=-1) / cm.sum(axis=(-2, -1))
    return (
        np.stack(
            [accuracy, precision.mean(axis=-1), recall.mean(axis=-1), f1.mean(axis=-1)], axis=-1
        ),
        precision,
        recall,
        f1,
    )


def read_predictions(relative, expected_ids, expected_y):
    path = ROOT / relative
    INPUTS[relative] = digest(path)
    pred, probs, annotations = [], [], Counter()
    with path.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            assert str(row["record_id"]) == expected_ids[i], (relative, i, "record order")
            assert row["expected"] == LABELS[int(expected_y[i])], (relative, i, "label order")
            pred.append(LABELS.index(row["diagnosis"]))
            probs.append([row["probabilities"][label] for label in LABELS])
            annotations["fallback"] += bool(row["used_fallback"])
            annotations["abstention"] += bool(row["abstained"])
            annotations["drivers"] += len(row["causal_drivers"])
            annotations["facts"] += len(row["retrieved_facts"])
            annotations["history_" + row["history"]["mode"]] += 1
            attribution = row["metadata"].get("causal_attribution", {})
            annotations["component_attribution"] += bool(
                attribution.get("component_available", attribution.get("available", False))
            )
            annotations["fused_attribution"] += bool(attribution.get("fused_available", False))
    probabilities = np.asarray(probs)
    assert len(pred) == len(expected_y)
    assert np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0)
    assert np.max(np.abs(probabilities.sum(axis=1) - 1)) < 1e-6
    assert np.array_equal(np.argmax(probabilities, axis=1), pred)
    return np.asarray(pred), probabilities, dict(annotations)


def savefig(name):
    plt.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.savefig(FIG / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close()


def classification(evidence):
    splits, patient_sets = {}, {}
    for split in ["train", "validation", "test"]:
        relative = "data/ptbxl-prepared/" + split + ".npz"
        path = ROOT / relative
        INPUTS[relative] = digest(path)
        with np.load(path, allow_pickle=False) as a:
            labels, patients, ids = (
                a["labels"],
                a["patient_ids"].astype(str),
                a["record_ids"].astype(str),
            )
            splits[split] = {
                "records": len(labels),
                "patients": len(set(patients)),
                "class_counts": np.bincount(labels, minlength=5).tolist(),
                "folds": sorted(set(a["folds"].tolist())),
            }
            patient_sets[split] = set(patients)
            if split == "test":
                y, patient_ids, record_ids = labels, patients, ids
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not patient_sets[a] & patient_sets[b]
    table(
        "cohorts",
        ["Split", "ECGs", "Patients", *LABELS],
        [[s.title(), v["records"], v["patients"], *v["class_counts"]] for s, v in splits.items()],
    )
    systems = [
        ("Previous repository", "outputs/ptbxl-advanced-ensemble-test.jsonl"),
        ("Released stack", "outputs/ptbxl-nextgen-ensemble-test.jsonl"),
        ("Waveform specialist", "outputs/ptbxl-improved-test.jsonl"),
    ]
    unique_patients, group = np.unique(patient_ids, return_inverse=True)
    patient_cms, results, predictions = [], {}, {}
    for name, relative in systems:
        pred, probs, annotations = read_predictions(relative, record_ids, y)
        pcm = np.zeros((len(unique_patients), 5, 5), dtype=np.int64)
        np.add.at(pcm, (group, y, pred), 1)
        cm = pcm.sum(axis=0)
        metrics, precision, recall, f1 = metric_arrays(cm)
        patient_cms.append(pcm.reshape(len(unique_patients), 25))
        results[name] = {
            "metrics": metrics.tolist(),
            "confusion_matrix": cm.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "annotations": annotations,
        }
        predictions[name] = pred
        if name == "Released stack":
            final_probs, final_cm = probs, cm
    rng = np.random.default_rng(SEED)
    draws = [[] for _ in systems]
    for start in range(0, RESAMPLES, 128):
        weights = rng.multinomial(
            len(unique_patients),
            np.full(len(unique_patients), 1 / len(unique_patients)),
            size=min(128, RESAMPLES - start),
        )
        for i, pcm in enumerate(patient_cms):
            cm = (weights @ pcm).reshape(-1, 5, 5)
            draws[i].append(metric_arrays(cm)[0])
    draws = [np.concatenate(parts) for parts in draws]
    for (name, _), samples in zip(systems, draws):
        results[name]["patient_cluster_ci95"] = np.quantile(
            samples, [0.025, 0.975], axis=0
        ).T.tolist()
    comparison = {}
    for i, comparator in [(0, "Previous repository"), (2, "Waveform specialist")]:
        delta = draws[1] - draws[i]
        comparison["Released minus " + comparator] = {
            "point": (
                np.array(results["Released stack"]["metrics"]) - results[comparator]["metrics"]
            ).tolist(),
            "ci95": np.quantile(delta, [0.025, 0.975], axis=0).T.tolist(),
        }
    table(
        "classification",
        ["System", "Accuracy", "Macro P", "Macro R", "Macro F1"],
        [[name, *[f"{100 * v:.2f}" for v in results[name]["metrics"]]] for name, _ in systems],
    )
    table(
        "classification_ci",
        ["Metric", "Estimate (%)", "Patient-cluster 95% CI (%)"],
        [
            [name, f"{100 * v:.2f}", f"{100 * ci[0]:.2f}--{100 * ci[1]:.2f}"]
            for name, v, ci in zip(
                ["Accuracy", "Macro precision", "Macro recall", "Macro F1"],
                results["Released stack"]["metrics"],
                results["Released stack"]["patient_cluster_ci95"],
            )
        ],
        "lrr",
    )
    final = results["Released stack"]
    table(
        "per_class",
        ["Class", "Support", "Predicted", "Correct", "P (%)", "R (%)", "F1 (%)"],
        [
            [
                label,
                int(final_cm[i].sum()),
                int(final_cm[:, i].sum()),
                int(final_cm[i, i]),
                *[
                    f"{100 * final[field][i]:.2f}"
                    for field in ["per_class_precision", "per_class_recall", "per_class_f1"]
                ],
            ]
            for i, label in enumerate(LABELS)
        ],
    )
    table(
        "paired",
        ["Released minus comparator", "Accuracy delta (pp)", "Macro F1 delta (pp)"],
        [
            [
                key.replace("Released minus ", ""),
                *[
                    f"{100 * v['point'][j]:+.2f} [{100 * v['ci95'][j][0]:+.2f}, {100 * v['ci95'][j][1]:+.2f}]"
                    for j in [0, 3]
                ],
            ]
            for key, v in comparison.items()
        ],
        "lrr",
    )
    confidence = final_probs.max(axis=1)
    correct = predictions["Released stack"] == y
    order = np.argsort(-confidence, kind="stable")
    coverages = np.arange(1, len(y) + 1) / len(y)
    selective_accuracy = np.cumsum(correct[order]) / np.arange(1, len(y) + 1)
    reliability = []
    for i in range(15):
        mask = (confidence >= i / 15) & (
            (confidence < (i + 1) / 15) if i < 14 else (confidence <= 1)
        )
        if mask.any():
            reliability.append(
                {
                    "bin": i,
                    "n": int(mask.sum()),
                    "confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    conf_metrics = {
        "log_loss": float(-np.log(np.clip(final_probs[np.arange(len(y)), y], 1e-15, 1)).mean()),
        "multiclass_brier": float(((final_probs - np.eye(5)[y]) ** 2).sum(axis=1).mean()),
        "ece_15_bin": float(
            sum(r["n"] / len(y) * abs(r["confidence"] - r["accuracy"]) for r in reliability)
        ),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "accuracy_at_80_percent_coverage": float(
            selective_accuracy[int(np.ceil(0.8 * len(y))) - 1]
        ),
        "retained_at_80_percent": int(np.ceil(0.8 * len(y))),
    }
    table(
        "calibration",
        ["Quantity", "Value"],
        [
            [k.replace("_", " "), f"{v:.6f}" if isinstance(v, float) else v]
            for k, v in conf_metrics.items()
        ],
        "lr",
    )
    errors = sorted(
        [(int(final_cm[i, j]), LABELS[i], LABELS[j]) for i in range(5) for j in range(5) if i != j],
        reverse=True,
    )
    evidence["classification"] = {
        "splits": splits,
        "test_unique_patients": len(unique_patients),
        "systems": results,
        "paired_differences": comparison,
        "confidence": conf_metrics,
        "reliability_bins": reliability,
        "largest_error_flows": errors[:10],
        "bootstrap": {
            "seed": SEED,
            "resamples": RESAMPLES,
            "estimand": "record-weighted metrics after resampling whole patient clusters",
            "interval": "percentile, pointwise, no model-selection uncertainty",
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in zip(
        axes,
        [final_cm, final_cm / final_cm.sum(axis=1, keepdims=True)],
        ["Counts", "Recall-normalized (%)"],
    ):
        ax.pcolormesh(
            np.arange(6) - 0.5,
            np.arange(6) - 0.5,
            mat,
            cmap="Blues",
            shading="flat",
            antialiased=False,
        )
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(4.5, -0.5)
        ax.set_aspect("equal")
        ax.set(
            xticks=range(5),
            yticks=range(5),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Predicted",
            ylabel="Reference",
            title=title,
        )
        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.0f}" if title == "Counts" else f"{100 * mat[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] > 0.55 * mat.max() else "#132d40",
                    fontsize=9,
                )
    fig.tight_layout()
    savefig("confusion")
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    x = np.arange(5)
    for offset, field, label, color in [
        (-0.24, "per_class_precision", "Precision", "#1c607e"),
        (0, "per_class_recall", "Recall", "#e49634"),
        (0.24, "per_class_f1", "F1", "#57a493"),
    ]:
        ax.bar(x + offset, np.array(final[field]) * 100, 0.24, label=label, color=color)
    ax.set(xticks=x, xticklabels=LABELS, ylabel="Percent", ylim=(0, 100))
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    savefig("per_class")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].scatter(
        [r["confidence"] for r in reliability],
        [r["accuracy"] for r in reliability],
        s=[max(12, r["n"] * 0.3) for r in reliability],
        color="#1c607e",
        alpha=0.85,
    )
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence (15 bins)",
        ylabel="Observed accuracy",
        title="Reliability; marker area reflects bin size",
    )
    axes[1].plot(coverages * 100, (1 - selective_accuracy) * 100, color="#1c607e")
    axes[1].set(
        xlim=(20, 100),
        ylim=(0, 35),
        xlabel="Retained records (%)",
        ylabel="Selective error (%)",
        title="Confidence-ranked selective risk",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig("calibration")
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    pos = np.arange(3)
    for j, offset, label, color in [
        (0, -0.18, "Accuracy", "#1c607e"),
        (3, 0.18, "Macro F1", "#e49634"),
    ]:
        vals = np.array([results[n]["metrics"][j] for n, _ in systems]) * 100
        cis = np.array([results[n]["patient_cluster_ci95"][j] for n, _ in systems]) * 100
        ax.errorbar(
            pos + offset,
            vals,
            fmt="o",
            markersize=6,
            label=label,
            color=color,
            yerr=np.array([vals - cis[:, 0], cis[:, 1] - vals]),
            capsize=3,
        )
    ax.set(
        xticks=pos,
        xticklabels=["Previous repository", "Released stack", "Specialist (not promoted)"],
        ylabel="Percent",
        ylim=(45, 85),
    )
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    savefig("system_comparison")


def model_and_data(evidence):
    history = read_json("artifacts/ptbxl-nextgen-final/training_history.json")
    graph = read_json("artifacts/ptbxl-nextgen-final/causal_graph.json")
    knowledge = read_json("artifacts/ptbxl-nextgen-final/knowledge_index.json")
    evidence["training"] = history
    evidence["graph"] = {
        "keys": list(graph),
        "network_keys": list(graph.get("network", {})),
        "diagnosis_parents": graph.get("network", {}).get("parents", {}).get("diagnosis", []),
        "edges": sum(len(v) for v in graph.get("network", {}).get("parents", {}).values()),
        "knowledge_documents": len(knowledge.get("documents", [])),
    }
    evidence["dataset_manifest"] = read_json("data/paper-prepared/manifest.json")
    evidence["local_dataset_manifest"] = read_json("data/paper-local-prepared/manifest.json")
    evidence["run_metadata"] = read_json("results/CARE_ECG_nextgen_run_metadata.json")
    evidence["publication_receipt"] = read_json("outputs/github-publish-result.json")
    relative = "results/CARE_ECG_nextgen_results.xlsx"
    INPUTS[relative] = digest(ROOT / relative)
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(ROOT / relative) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(s.itertext()) for s in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        summary = {}
        for row in sheet.findall(".//s:sheetData/s:row", ns):
            values = []
            for cell in row:
                kind = cell.attrib.get("t")
                value = cell.find("s:v", ns)
                if kind == "inlineStr":
                    values.append("".join(cell.find("s:is", ns).itertext()))
                elif kind == "s":
                    values.append(strings[int(value.text)])
                elif value is not None:
                    try:
                        values.append(float(value.text))
                    except ValueError:
                        values.append(value.text)
                else:
                    values.append(None)
            if len(values) > 1:
                summary[str(values[0])] = values[1]
        evidence["workbook_summary"] = summary
    assert summary["Prediction records"] == 2158 and summary["scp_codes_per_report"] == 1
    evidence["selection_lock"] = read_json("results/CARE_ECG_nextgen_locked_selection.json")
    evidence["specialist"] = read_json("results/CARE_ECG_improved_run_metadata.json")
    search = read_json("results/CARE_ECG_validation_meta_search.json")
    candidates = []
    for row in search["candidate_results"]:
        result = row.get("best_additive_patient_group_oof", {})
        candidates.append(
            {
                "name": row["name"],
                "standalone": row.get("standalone_metrics"),
                "additive": result.get("metrics"),
            }
        )
    evidence["meta_search"] = {
        "candidate_count": search["candidate_count"],
        "candidates": candidates,
        "decision": search["decision"],
    }
    table(
        "meta_candidates",
        ["Candidate", "Standalone F1 (%)", "Added-stack F1 (%)"],
        [
            [
                r["name"],
                f"{r['standalone']['macro_f1'] * 100:.2f}" if r["standalone"] else "--",
                f"{r['additive']['macro_f1'] * 100:.2f}" if r["additive"] else "--",
            ]
            for r in candidates
        ],
        "lrr",
    )
    fusion = history["diagnostics"]["probability_fusion"]["candidates"]
    table(
        "fusion",
        ["Neural weight", "Accuracy (%)", "Macro P (%)", "Macro R (%)", "Macro F1 (%)"],
        [
            [
                r["neural_weight"],
                *[
                    f"{100 * r[k]:.2f}"
                    for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
                ],
            ]
            for r in fusion
        ],
    )
    stack = read_json(
        "artifacts/ptbxl-benchmark/stack-three-source/probability_stacker_validation_report.json"
    )
    grid = stack["candidate_search"]
    evidence["stack_search"] = {
        "candidate_count": len(grid),
        "regularization": sorted({r["regularization"] for r in grid}),
        "class_weight_power": sorted({r["class_weight_power"] for r in grid}),
    }
    rows = evidence["dataset_manifest"]["datasets"][0]["tasks"]
    table(
        "qa_tasks",
        ["Released source task", "Rows"],
        [[k, v] for k, v in rows.items()],
        "p{0.78\\linewidth}r",
    )
    epochs = np.arange(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for split, color in [("train", "#1c607e"), ("validation", "#e49634")]:
        axes[0].plot(
            epochs, [r["macro_f1"] * 100 for r in history[split]], label=split.title(), color=color
        )
        axes[1].plot(
            epochs, [r["reconstruction"] for r in history[split]], label=split.title(), color=color
        )
    for ax in axes:
        ax.axvline(history["diagnostics"]["best_epoch"], ls="--", color="grey", lw=1)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Macro F1 (%)")
    axes[1].set_ylabel("Reconstruction MSE")
    fig.tight_layout()
    savefig("training")
    regs = evidence["stack_search"]["regularization"]
    powers = evidence["stack_search"]["class_weight_power"]
    mat = np.full((len(regs), len(powers)), np.nan)
    for r in grid:
        mat[regs.index(r["regularization"]), powers.index(r["class_weight_power"])] = (
            100 * r["metrics"]["macro_f1"]
        )
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    im = ax.pcolormesh(
        np.arange(len(powers) + 1) - 0.5,
        np.arange(len(regs) + 1) - 0.5,
        mat,
        cmap="YlGnBu",
        shading="flat",
        antialiased=False,
    )
    ax.set_xlim(-0.5, len(powers) - 0.5)
    ax.set_ylim(len(regs) - 0.5, -0.5)
    ax.set(
        xticks=range(len(powers)),
        yticks=range(len(regs)),
        xticklabels=powers,
        yticklabels=regs,
        xlabel="Class-weight exponent",
        ylabel="L2 penalty coefficient",
    )
    for i in range(len(regs)):
        for j in range(len(powers)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if mat[i, j] > 68.5 else "#132d40",
            )
    fig.colorbar(im, ax=ax, label="Fold-9 selection macro F1 (%)")
    savefig("stack_search")


def security(evidence):
    studies = {}
    for label, directory in [
        ("validation", "security-final-validation-c3-b1-b3-s3-20261026"),
        ("heldout", "security-final-heldout-c3-b1-b3-s3-20261026"),
        ("momentum_validation", "security-mipgd-validation-20261026"),
    ]:
        d = read_json("results/" + directory + "/security_results.json")
        rows = []
        obs_groups = {}
        totals = Counter()
        for attack, arms in d["e1"].items():
            for constraint, cell in arms.items():
                for stratum, summary in cell["summary"].items():
                    if "B=" not in stratum:
                        continue
                    budget = int(stratum.split("B=")[1].split("|")[0])
                    a = summary["automated_robustness"]
                    obs = [o for o in cell["observations"] if o["budget"] == budget]
                    assert len(obs) == 30 and len({o["patient_id"] for o in obs}) == 10
                    fields = {
                        "targeted_diagnosis_success": "targeted_diagnosis_attack_success_rate",
                        "untargeted_diagnosis_success": "untargeted_diagnosis_attack_success_rate",
                        "released_target_claim_success": "released_target_claim_attack_success_rate",
                        "robust_correct": "robust_accuracy",
                        "selected_constraint_satisfied": "valid_selected_perturbation_rate",
                    }
                    for field, key in fields.items():
                        assert np.isclose(
                            np.mean([o["metrics"]["automated_robustness"][field] for o in obs]),
                            a[key]["estimate"],
                        )
                    row = {
                        "attack": attack,
                        "constraint": constraint,
                        "budget": budget,
                        "n": len(obs),
                        "patients": 10,
                        "targeted": a["targeted_diagnosis_attack_success_rate"],
                        "untargeted": a["untargeted_diagnosis_attack_success_rate"],
                        "released_claim": a["released_target_claim_attack_success_rate"],
                        "robust_accuracy": a["robust_accuracy"],
                        "valid": a["valid_selected_perturbation_rate"],
                        "distortion": a["distortion_all_valid_selected"],
                        "queries": sum(o["queries_used"] for o in obs),
                        "primary_evaluable": summary["primary_endpoint_evaluable_observations"],
                    }
                    rows.append(row)
                    obs_groups[(attack, constraint, budget)] = obs
                    totals["observations"] += len(obs)
                    totals["queries"] += row["queries"]
        short = {"f1_pgd": "F1 PGD", "f1_cf": "F1-CF", "f3_k2": "F3 K2", "f3_k3": "F3 K3"}
        pct = lambda x: (
            f"{100 * x['estimate']:.1f} [{100 * x['lower']:.1f}, {100 * x['upper']:.1f}]"
        )
        table(
            "security_" + label,
            [
                "Attack",
                "Constraint",
                "B",
                "Target ASR (%) [95% CI]",
                "Untargeted (%)",
                "Robust (%)",
            ],
            [
                [
                    short[r["attack"]],
                    r["constraint"].replace("s_inf", "L-infinity").replace("s_spec", "Spectral"),
                    r["budget"],
                    pct(r["targeted"]),
                    f"{100 * r['untargeted']['estimate']:.1f}",
                    f"{100 * r['robust_accuracy']['estimate']:.1f}",
                ]
                for r in rows
            ],
            "llrrrr",
        )
        table(
            "distortion_" + label,
            ["Attack", "Constraint", "B", "RMSE (mV)", "PRD (%)", "SNR (dB)", "Queries"],
            [
                [
                    short[r["attack"]],
                    r["constraint"],
                    r["budget"],
                    *[
                        f"{r['distortion'][k]['mean']:.4f}"
                        for k in ["rmse_mv", "prd_percent", "snr_db"]
                    ],
                    r["queries"],
                ]
                for r in rows
            ],
            "llrrrrr",
        )
        macro = {
            str(b): {
                field: float(np.mean([r[field]["estimate"] for r in rows if r["budget"] == b]))
                for field in [
                    "targeted",
                    "untargeted",
                    "released_claim",
                    "robust_accuracy",
                    "valid",
                ]
            }
            for b in [1, 3]
        }
        paired = []
        for attack in ["f3_k2", "f3_k3"]:
            for constraint in ["s_inf", "s_spec"]:
                for budget in [1, 3]:
                    left = obs_groups[(attack, constraint, budget)]
                    right = obs_groups[("f1_pgd", constraint, budget)]
                    key = lambda o: (o["patient_id"], o["record_id"], o["seed"], o["budget"])
                    reference = {key(o): o for o in right}
                    patient_deltas = defaultdict(list)
                    for o in left:
                        r = reference[key(o)]
                        a = o["metrics"]["automated_robustness"]
                        b = r["metrics"]["automated_robustness"]
                        patient_deltas[o["patient_id"]].append(
                            [
                                float(a[f]) - float(b[f])
                                for f in [
                                    "targeted_diagnosis_success",
                                    "untargeted_diagnosis_success",
                                    "released_target_claim_success",
                                ]
                            ]
                        )
                    values = np.array(
                        [np.mean(v, axis=0) for _, v in sorted(patient_deltas.items())]
                    )
                    rng = np.random.default_rng(SEED)
                    samples = values[
                        rng.integers(0, len(values), size=(RESAMPLES, len(values)))
                    ].mean(axis=1)
                    paired.append(
                        {
                            "attack": attack,
                            "constraint": constraint,
                            "budget": budget,
                            "point": values.mean(axis=0).tolist(),
                            "ci95": np.quantile(samples, [0.025, 0.975], axis=0).T.tolist(),
                        }
                    )
        studies[label] = {
            "protocol": {k: v for k, v in d["protocol"].items() if k != "validation_lock"},
            "e0": d["e0"]["summary"],
            "victim_scope": d["artifacts"].get("victim_scope"),
            "admissibility": d["admissibility"],
            "events": d["event_log"]["verification"],
            "cells": rows,
            "totals": dict(totals),
            "macro": macro,
            "paired_auto_reanalysis": paired,
            "limitations": d["limitations"],
        }
        if label == "heldout":
            table(
                "security_paired",
                ["Adaptive", "Constraint", "B", "Target delta (pp)", "Untargeted delta (pp)"],
                [
                    [
                        short[r["attack"]],
                        r["constraint"],
                        r["budget"],
                        *[
                            f"{100 * r['point'][j]:+.1f} [{100 * r['ci95'][j][0]:+.1f}, {100 * r['ci95'][j][1]:+.1f}]"
                            for j in [0, 1]
                        ],
                    ]
                    for r in paired
                ],
                "llrrr",
            )
    evidence["security"] = studies
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, budget in zip(axes, [1, 3]):
        rows = [r for r in studies["heldout"]["cells"] if r["budget"] == budget]
        for offset, constraint, color, label in [
            (-0.17, "s_inf", "#1c607e", "L-infinity"),
            (0.17, "s_spec", "#e49634", "Spectral"),
        ]:
            selected = [r for r in rows if r["constraint"] == constraint]
            values = np.array([r["targeted"]["estimate"] for r in selected]) * 100
            lower = np.array([r["targeted"]["lower"] for r in selected]) * 100
            upper = np.array([r["targeted"]["upper"] for r in selected]) * 100
            ax.bar(
                np.arange(4) + offset,
                values,
                0.33,
                color=color,
                label=label,
                yerr=[values - lower, upper - values],
                capsize=3,
            )
        ax.set(
            xticks=range(4),
            xticklabels=["PGD", "CF", "K2", "K3"],
            ylim=(0, 85),
            ylabel="Targeted ASR (%)",
            title=f"Complete-system budget B={budget}",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    savefig("security")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    evidence = {
        "report_date": "2026-10-28",
        "snapshot_time": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "analysis_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "analysis_scope": "Retrospective aggregate reanalysis of existing saved artifacts. No retraining or hyperparameter selection.",
    }
    classification(evidence)
    model_and_data(evidence)
    security(evidence)
    from build_extensions import extend

    extend(evidence, ROOT, OUT, INPUTS)
    metadata = evidence["run_metadata"]
    for key, relative in [
        ("care_model", "artifacts/ptbxl-nextgen-final/causal_vae.pt"),
        (
            "locked_stacker",
            "artifacts/ptbxl-benchmark/stack-three-source/locked_probability_stacker.npz",
        ),
        ("selection_lock", "results/CARE_ECG_nextgen_locked_selection.json"),
        ("predictions", "outputs/ptbxl-nextgen-ensemble-test.jsonl"),
    ]:
        observed = digest(ROOT / relative)
        INPUTS[relative] = observed
        assert observed == metadata["sha256"][key], (key, "hash mismatch")
    for i, key in enumerate(["accuracy", "macro_precision", "macro_recall", "macro_f1"]):
        assert np.isclose(
            evidence["classification"]["systems"]["Released stack"]["metrics"][i],
            metadata["held_out_results"]["direct_and_first_mention_text_to_scp"][key],
        )
    evidence["input_sha256"] = dict(sorted(INPUTS.items()))
    evidence["current_source_sha256"] = {
        str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src/care_ecg").rglob("*.py"))
    }
    selected = [
        "artifacts/ptbxl-nextgen-final/causal_vae.pt",
        "artifacts/ptbxl-benchmark/stack-three-source/locked_probability_stacker.npz",
        "results/CARE_ECG_nextgen_locked_selection.json",
        "outputs/ptbxl-nextgen-ensemble-test.jsonl",
        "data/ptbxl-prepared/test.npz",
        "results/security-final-validation-c3-b1-b3-s3-20261026/security_results.json",
        "results/security-final-heldout-c3-b1-b3-s3-20261026/security_results.json",
        "results/security-mipgd-validation-20261026/security_results.json",
        "data/paper-prepared/manifest.json",
    ]
    ledger = [
        r"\begin{longtable}{p{0.55\linewidth}p{0.38\linewidth}}",
        r"\toprule",
        r"Input artifact & SHA-256 (two consecutive halves) \\",
        r"\midrule\endhead",
    ]
    for relative in selected:
        value = INPUTS[relative]
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
    (TAB / "provenance.tex").write_text("\n".join(ledger) + "\n")
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "classification": evidence["classification"]["systems"]["Released stack"],
                "confidence": evidence["classification"]["confidence"],
                "pairs": evidence["classification"]["paired_differences"],
                "graph": evidence["graph"],
                "security": {
                    k: {"totals": v["totals"], "macro": v["macro"]}
                    for k, v in evidence["security"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
