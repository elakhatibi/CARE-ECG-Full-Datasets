"""Batch-bounded inference over full waveform and QA indexes."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import CAREConfig
from .data import LEAD_ORDER, preprocess_ecg
from .paper_data import _relative_waveform, sha256_file
from .pipeline import CAREECGPipeline
from .verification import hallucination_risk, unsupported_claim_risk


def read_index(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_waveform(row: dict, root: Path, config: CAREConfig) -> np.ndarray:
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Waveform loading requires care-ecg[data]") from exc
    relative = _relative_waveform(root, row["waveform_path"])
    values, fields = wfdb.rdsamp(str(root / relative))
    names = [name.upper() for name in fields["sig_name"]]
    required = [name.upper() for name in LEAD_ORDER]
    if len(names) != 12 or set(names) != set(required):
        raise ValueError(f"Expected the 12 standard leads in {relative}, got {names}")
    units = fields["units"]
    scales = {"mv": 1.0, "uv": 0.001, "µv": 0.001, "μv": 0.001, "v": 1000.0}
    channels = []
    for name in required:
        index = names.index(name)
        unit = units[index].lower()
        if unit not in scales:
            raise ValueError(f"Unsupported waveform units: {units[index]}")
        channels.append(values[:, index] * scales[unit])
    return preprocess_ecg(
        np.asarray(channels, dtype=np.float32),
        int(fields["fs"]),
        config.data.sample_rate,
        config.data.seconds,
        config.data.leads,
        "none" if config.data.normalization == "global" else config.data.normalization,
    )


def _load_pipeline(args):
    config = CAREConfig.from_yaml(Path(args.artifacts) / "config.yaml")
    if args.generator:
        config.generation.backend = args.generator
    if args.model:
        config.generation.model = args.model
    return CAREECGPipeline.load(args.artifacts, config)


def predict_waveform_index(args) -> int:
    """Stream all source ECGs without requiring classifier-compatible labels."""
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    pipeline = _load_pipeline(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    folds = set(args.folds) if args.folds else None
    records = 0
    patients: set[str] = set()
    batch: list[dict] = []
    with output.open("x", encoding="utf-8") as handle:

        def flush():
            nonlocal records
            signals = np.stack(
                [load_waveform(row, Path(args.root), pipeline.config) for row in batch]
            )
            latent, neural = pipeline.encode_batch_with_neural(signals, batch_size=args.batch_size)
            for index, row in enumerate(batch):
                result = pipeline.run_encoded(
                    latent[index],
                    args.query,
                    patient_id=row["patient_id"],
                    neural_probabilities=None if neural is None else neural[index],
                ).to_dict()
                result.update(
                    {
                        "dataset": row["dataset"],
                        "record_id": row["record_id"],
                        "patient_id": row["patient_id"],
                        "source_metadata": row,
                    }
                )
                result["metadata"]["evaluation_scope"] = (
                    "source-fold subset"
                    if folds
                    else "full-corpus inference; includes training records; not a held-out estimate"
                )
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                records += 1
                patients.add(row["patient_id"])
            handle.flush()
            print(f"predicted {records} records", flush=True)
            batch.clear()

        for row in read_index(args.index):
            if folds and row.get("fold") not in folds:
                continue
            batch.append(row)
            if len(batch) >= args.batch_size:
                flush()
        if batch:
            flush()
    summary = {
        "records": records,
        "patients": len(patients),
        "index_sha256": sha256_file(Path(args.index)),
        "folds": sorted(folds) if folds else None,
        "patient_or_record_caps": None,
        "completed": True,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return 0


@contextmanager
def waveform_lookup(index, directory):
    """Keep large MIMIC joins on disk instead of retaining 800k metadata dicts."""
    if index is None:
        yield lambda _: None
        return
    with tempfile.TemporaryDirectory(prefix="care-qa-join-", dir=directory) as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "studies.sqlite"))
        try:
            connection.execute("CREATE TABLE studies (id TEXT PRIMARY KEY, payload TEXT)")
            connection.executemany(
                "INSERT INTO studies VALUES (?, ?)",
                ((row["record_id"], json.dumps(row)) for row in read_index(index)),
            )
            connection.commit()

            def lookup(record_id):
                row = connection.execute(
                    "SELECT payload FROM studies WHERE id = ?", (record_id,)
                ).fetchone()
                return json.loads(row[0]) if row else None

            yield lookup
        finally:
            connection.close()


def answer_qa_index(args) -> int:
    """Process every QA source row without feeding reference answers to generation.

    Source rows with no linked WFDB record use the text retrieval/generation path;
    they never receive invented waveforms, biomarkers, or causal attributions.
    """
    pipeline = _load_pipeline(args)
    if args.mimic_index and not args.mimic_root:
        raise ValueError("--mimic-root is required with --mimic-index")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = waveform_count = 0
    with (
        waveform_lookup(args.mimic_index, output.parent) as lookup,
        output.open("x", encoding="utf-8") as handle,
    ):
        for row in read_index(args.index):
            query = row["query"]
            waveform = lookup(row.get("study_id"))
            if waveform is not None:
                if str(waveform["patient_id"]) != str(row["patient_id"]):
                    raise ValueError("QA patient ID does not match the linked ECG")
                signal = load_waveform(waveform, Path(args.mimic_root), pipeline.config)
                result = pipeline.run(signal, query, patient_id=row["patient_id"]).to_dict()
                result["input_mode"] = "linked_mimic_waveform"
                waveform_count += 1
            else:
                context = row.get("context", "")
                if not context and row.get("study_id"):
                    # Source ECG reports are inputs, never the Q-field target answer.
                    context = "\n".join(
                        str(value) for key, value in row["raw"].items() if key.startswith("report_")
                    )
                retrieval_query = query + ("\n" + context if context else "")
                facts = pipeline.retriever.retrieve(
                    retrieval_query,
                    (),
                    {},
                    pipeline.config.retrieval.top_k,
                    pipeline.config.retrieval.min_score,
                )
                evidence = "\n".join(f"[Fact {i}] {fact.text}" for i, fact in enumerate(facts, 1))
                prompt = (
                    f"Patient Query: {query}\nClinical Context: {context}\n"
                    f"Retrieved Medical Facts (RAG):\n{evidence}\n"
                    "Answer the question using the context and retrieved facts. "
                    "Cite factual claims with [Fact n]. No waveform was supplied."
                )
                agent = pipeline.response_agent
                answer = agent.generator.generate(prompt, facts)
                risk = hallucination_risk(answer, facts, agent.fuzzy_match_threshold)
                unsupported = unsupported_claim_risk(answer, facts, agent.fuzzy_match_threshold)
                fallback = agent.fallback_to_rag_only and (
                    risk > agent.hallucination_threshold or unsupported > 0
                )
                if fallback:
                    answer = (
                        "\n".join(f"{fact.text} [Fact {i}]" for i, fact in enumerate(facts, 1))
                        or "Insufficient retrieved evidence for a grounded answer."
                    )
                result = {
                    "query": query,
                    "rationale": answer,
                    "retrieved_facts": [asdict(fact) for fact in facts],
                    "input_mode": "text_only",
                    "used_fallback": fallback,
                    "hallucination_score": hallucination_risk(
                        answer, facts, agent.fuzzy_match_threshold
                    ),
                    "unsupported_claim_score": unsupported_claim_risk(
                        answer, facts, agent.fuzzy_match_threshold
                    ),
                }
            result.update(
                {
                    "record_id": row["record_id"],
                    "patient_id": row["patient_id"],
                    "task": row["task"],
                    "reference_answer": row["reference_answer"],
                    "evaluation_status": "requires task-specific adjudication",
                }
            )
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            if count % 100 == 0:
                handle.flush()
                print(f"answered {count} QA source rows", flush=True)
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(
            {
                "records": count,
                "waveform_records": waveform_count,
                "text_only_records": count - waveform_count,
                "patient_or_record_caps": None,
                "index_sha256": sha256_file(Path(args.index)),
                "completed": True,
                "paper_comparable_metrics": False,
            },
            indent=2,
        )
        + "\n"
    )
    return 0
