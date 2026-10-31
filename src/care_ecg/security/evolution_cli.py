"""CLI for training and validating the bounded two-policy evolution loop."""

from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from ..data import load_npz
from ..pipeline import CAREECGPipeline
from .contracts import DefenseKind
from .defenses import ShippedVerifier
from .evolution import EvolutionConfig, evolve
from .learned import DPOVerifier
from .victim import InternalCAREECGVictim


def _run(args):
    # Imported at execution to keep CLI parser construction free of cycles.
    import json

    from ..cli import (
        _directory_identity,
        _security_code_identity,
        _security_cohort_cases,
        _security_cohort_manifest,
        _security_constraint,
        _security_raw_signals,
        _sha256_file,
    )

    raw = yaml.safe_load(Path(args.evolution_config).read_text())
    if raw.get("schema_version") != "care-ecg-evolution-config.v1":
        raise ValueError("Unsupported evolution config schema")
    config_values = dict(raw.get("evolution", {}))
    for field in ("rounds", "steps", "dpo_epochs"):
        if getattr(args, field) is not None:
            config_values[field] = getattr(args, field)
    config = EvolutionConfig(**config_values)
    if Path(args.output).resolve().is_relative_to(Path(args.artifacts).resolve()):
        raise ValueError("Evolution output must be outside the immutable victim artifact directory")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    pipeline = CAREECGPipeline.load(args.artifacts)
    training = load_npz(args.training_dataset, require_labels=True)
    validation = load_npz(args.validation_dataset, require_labels=True)
    for bundle, allowed in [(training, set(range(1, 9))), (validation, {9})]:
        if bundle.patient_ids is None or bundle.record_ids is None or bundle.folds is None:
            raise ValueError("Evolution requires patient, record, and fold metadata")
        if not set(bundle.folds.tolist()) <= allowed:
            raise ValueError("Evolution refuses fold 10 or an invalid training/validation split")
    if set(training.patient_ids) & set(validation.patient_ids):
        raise ValueError("Full training/validation patient sets overlap")
    protocol = raw.get("protocol", {})
    if protocol.get("records_per_patient", "all") not in {"all", "first"}:
        raise ValueError("records_per_patient must be all or first")
    targets = (
        tuple(args.targets.split(","))
        if args.targets
        else tuple(protocol.get("targets", ("T1", "T2", "T3", "T4")))
    )

    def cases(bundle, limit):
        return _security_cohort_cases(
            bundle,
            _security_raw_signals(bundle.signals, pipeline, None),
            pipeline.causal_model.label_names,
            targets,
            max_candidates_per_target=limit,
            all_records=protocol.get("records_per_patient", "all") == "all",
        )

    train_limit = args.max_train_patients_per_target
    if train_limit is None:
        train_limit = protocol.get("max_train_patients_per_target")
    validation_limit = args.max_validation_patients_per_target
    if validation_limit is None:
        validation_limit = protocol.get("max_validation_patients_per_target")
    training_cases, validation_cases = (
        cases(training, train_limit),
        cases(validation, validation_limit),
    )
    values = dict(raw["constraint"])
    name = values.pop("name")
    constraint = _security_constraint(name, values, float(pipeline.config.data.sample_rate))

    def provider(*, case, verifier_policy):
        gate = DPOVerifier(verifier_policy, ShippedVerifier())
        return InternalCAREECGVictim(
            pipeline,
            patient_id=case.patient_id,
            defense_evaluators={DefenseKind.V1: gate},
            defense_surrogates={DefenseKind.V1: gate.surrogate},
        )

    provenance = {
        "model": _directory_identity(args.artifacts),
        "code": _security_code_identity(),
        "config_sha256": _sha256_file(args.evolution_config),
        "training_dataset_sha256": _sha256_file(args.training_dataset),
        "validation_dataset_sha256": _sha256_file(args.validation_dataset),
        "training_cohort": _security_cohort_manifest(training_cases),
        "validation_cohort": _security_cohort_manifest(validation_cases),
        "constraint": {"name": name, **values},
        "targets": targets,
        "records_per_patient": protocol.get("records_per_patient", "all"),
        "max_train_patients_per_target": train_limit,
        "max_validation_patients_per_target": validation_limit,
        "execution_device": str(pipeline.device),
        "threads": args.threads,
    }
    result = evolve(
        training_cases,
        validation_cases,
        label_order=pipeline.causal_model.label_names,
        provider=provider,
        constraint=constraint,
        output=args.output,
        config=config,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "complete": result["complete"],
                "rounds": len(result["rounds"]),
                "config": asdict(config),
                "attacker_promotions": sum(r["attacker_promoted"] for r in result["rounds"]),
                "verifier_promotions": sum(r["verifier_promoted"] for r in result["rounds"]),
            },
            indent=2,
        )
    )
    return 0


def register_parser(subparsers):
    command = subparsers.add_parser(
        "security-evolve", help="co-evolve attacker/verifier action policies with DPO"
    )
    command.add_argument("artifacts")
    command.add_argument("training_dataset", help="raw-mV labeled NPZ, folds 1-8")
    command.add_argument("validation_dataset", help="separate patients in labeled fold 9 NPZ")
    command.add_argument("output", help="new versioned policy artifact directory")
    command.add_argument("--evolution-config", default="config/security_evolution.yaml")
    command.add_argument("--rounds", type=int)
    command.add_argument("--steps", type=int)
    command.add_argument("--dpo-epochs", type=int)
    command.add_argument("--threads", type=int, default=2)
    command.add_argument("--targets", help="explicit comma-separated development target subset")
    command.add_argument("--max-train-patients-per-target", type=int)
    command.add_argument("--max-validation-patients-per-target", type=int)
    command.set_defaults(func=_run)
