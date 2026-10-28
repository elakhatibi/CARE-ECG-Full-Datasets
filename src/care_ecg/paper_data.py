"""Versioned acquisition and streaming indexes for the three CARE-ECG datasets.

Waveforms stay in WFDB files. Indexes retain every source row and repeated patient;
training and held-out splits remain separate from full-corpus inference.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import http.client
import json
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zlib import crc32

QA_REVISION = "4fcb46974eed422532772caa483a5e621f006ff1"
PAPER_DATASETS = {
    "ptbxl": {
        "directory": "ptb-xl-1.0.1",
        "version": "1.0.1",
        "source": "https://physionet.org/content/ptb-xl/1.0.1/",
        "archive": "https://physionet.org/content/ptb-xl/get-zip/1.0.1/",
        "paper_records": 21837,
        "paper_patients": 18885,
    },
    "mimic": {
        "directory": "mimic-iv-ecg-1.0",
        "version": "1.0",
        "source": "https://physionet.org/content/mimic-iv-ecg/1.0/",
        "archive": "https://physionet.org/content/mimic-iv-ecg/get-zip/1.0/",
        "paper_records": 800000,
        "paper_patients": 160000,
        "paper_counts_approximate": True,
    },
    "expert-qa": {
        "directory": "ecg-expert-qa",
        "version": QA_REVISION,
        "source": "https://github.com/Zaozzz/ECG-Expert-QA",
        "archive": f"https://codeload.github.com/Zaozzz/ECG-Expert-QA/zip/{QA_REVISION}",
        "paper_records": 47211,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, retries: int = 5) -> None:
    """Resume partial transfers; retry limits count failures without forward progress."""
    if retries < 1:
        raise ValueError("retries must be positive")
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    failures_without_progress = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "care-ecg-reproduction/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url + ("&" if "?" in url else "?") + f"care_ecg_offset={offset}",
                    headers=headers,
                ),
                timeout=60,
            ) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError("Server returned a mismatched download range")
                expected = response.headers.get("Content-Length")
                received = 0
                last_progress = time.monotonic()
                with partial.open("ab" if resumed else "wb") as handle:
                    while chunk := response.read(64 * 1024):
                        handle.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_progress >= 30:
                            print(
                                f"{destination.name}: {handle.tell() / 1e9:.3f} GB",
                                file=sys.stderr,
                                flush=True,
                            )
                            last_progress = time.monotonic()
                if expected is not None and received != int(expected):
                    raise OSError("Incomplete HTTP download")
                if resumed:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers["Content-Range"]
                    )
                    if match is None or partial.stat().st_size != int(match.group(3)):
                        raise OSError("Partial HTTP range did not complete the archive")
            partial.replace(destination)
            return
        except (OSError, ValueError, http.client.HTTPException) as exc:
            current_size = partial.stat().st_size if partial.exists() else 0
            failures_without_progress = (
                0 if current_size > offset else failures_without_progress + 1
            )
            if failures_without_progress >= retries:
                raise
            print(
                f"{destination.name}: retrying from {current_size} bytes after {exc}; "
                f"{failures_without_progress}/{retries} failures without progress",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(2 ** max(0, failures_without_progress - 1), 10))


def extract_archive(archive: Path, destination: Path) -> None:
    """Strip release wrappers; reject traversal and symlink entries."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        files = [entry for entry in source.infolist() if not entry.is_dir()]
        roots = {PurePosixPath(entry.filename).parts[0] for entry in files}
        strip_root = len(roots) == 1 and all(
            len(PurePosixPath(entry.filename).parts) > 1 for entry in files
        )
        for entry in files:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlinks are unsupported: {entry.filename}")
            parts = path.parts[1:] if strip_root else path.parts
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Archive path escapes destination: {entry.filename}")
            if target.is_file() and target.stat().st_size == entry.file_size:
                checksum = 0
                with target.open("rb") as existing:
                    for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                        checksum = crc32(chunk, checksum)
                if checksum == entry.CRC:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with source.open(entry) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            temporary.replace(target)


def verify_checksums(root: Path) -> dict[str, Any]:
    """Verify every file listed by PhysioNet, including both PTB-XL rates."""
    checksum_path = root / "SHA256SUMS.txt"
    verified = 0
    with checksum_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            expected, name = line.rstrip().split(maxsplit=1)
            relative = PurePosixPath(name.lstrip("*"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe checksum path")
            path = root.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Checksum path escapes dataset root")
            if sha256_file(path) != expected:
                raise ValueError(f"Checksum mismatch: {path}")
            verified += 1
    if not verified:
        raise ValueError("Publisher checksum manifest is empty")
    return {"verified_files": verified, "manifest_sha256": sha256_file(checksum_path)}


@contextmanager
def _acquisition_lock(path: Path):
    """Prevent concurrent CLI runs from writing the same partial files on POSIX."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OSError(f"Another acquisition is already running: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def acquire_dataset(dataset: str, data_root: Path) -> Path:
    name = PAPER_DATASETS[dataset]["directory"]
    with _acquisition_lock(data_root / "downloads" / f"{name}.lock"):
        return _acquire_dataset(dataset, data_root)


def _acquire_dataset(dataset: str, data_root: Path) -> Path:
    spec = PAPER_DATASETS[dataset]
    root = data_root / str(spec["directory"])
    receipt_path = root / "download_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("version") == spec["version"] and receipt.get("complete") is True:
            return root
    archive = data_root / "downloads" / f"{spec['directory']}.zip"
    download_file(str(spec["archive"]), archive)
    extract_archive(archive, root)
    verification = (
        verify_checksums(root)
        if dataset != "expert-qa"
        else {
            "archive_sha256": sha256_file(archive),
            "revision": QA_REVISION,
        }
    )
    receipt = {**spec, "complete": True, "verification": verification}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return root


def _relative_waveform(root: Path, value: str) -> str:
    path = root / value
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Waveform path escapes dataset root: {value}")
    return value


def iter_waveform_metadata(
    dataset: str, root: str | Path, *, sampling_rate: int = 500
) -> Iterator[dict[str, Any]]:
    """Yield every metadata row, including unlabeled ECGs and repeated patients."""
    root = Path(root)
    if dataset == "ptbxl":
        if sampling_rate not in {100, 500}:
            raise ValueError("PTB-XL sampling_rate must be 100 or 500")
        path = root / "ptbxl_database.csv"
    elif dataset == "mimic":
        path = root / "record_list.csv"
    else:
        raise ValueError("Waveform dataset must be ptbxl or mimic")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if dataset == "ptbxl":
                value = {
                    "dataset": dataset,
                    "patient_id": str(row["patient_id"]),
                    "record_id": str(row["ecg_id"]),
                    "waveform_path": row["filename_hr" if sampling_rate == 500 else "filename_lr"],
                    "sampling_rate": sampling_rate,
                    "fold": int(row["strat_fold"]),
                    "scp_codes": ast.literal_eval(row["scp_codes"]),
                    "timestamp": row.get("recording_date"),
                }
            else:
                value = {
                    "dataset": dataset,
                    "patient_id": str(row["subject_id"]),
                    "record_id": str(row["study_id"]),
                    "waveform_path": row["path"],
                    "sampling_rate": 500,
                    "timestamp": row["ecg_time"],
                }
            value["waveform_path"] = _relative_waveform(root, value["waveform_path"])
            yield value


def _source_questions(payload: Any) -> list[str]:
    """Extract question fields from nested task payloads without target answers."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            try:
                payload = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                return []
    if isinstance(payload, list):
        return [question for item in payload for question in _source_questions(item)]
    if isinstance(payload, dict):
        questions = []
        for key, value in payload.items():
            if key in {"Q", "Question"} and isinstance(value, str):
                questions.append(value)
            elif key not in {"A", "Answer"} and isinstance(value, (dict, list)):
                questions.extend(_source_questions(value))
        return questions
    return []


def iter_expert_qa(root: str | Path) -> Iterator[dict[str, Any]]:
    """Retain every released task row and the complete original target payload.

    Nested dialogue targets stay intact. Patient IDs are never synthesized.
    The cross-modal source file uses A for questions and Q for answers.
    """
    for path in sorted(Path(root).glob("*.json")):
        if path.name in {"download_receipt.json", "manifest.json"}:
            continue
        with path.open(encoding="utf-8-sig") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise TypeError(f"Expected a JSON list in {path}")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object in {path}:{index}")
            cross_modal = path.name.startswith("Cross modal diagnosis")
            question_key, answer_key = ("A", "Q") if cross_modal else ("Q", "A")
            query = row.get(question_key, row.get("Question"))
            reference = row.get(answer_key, row.get("Answer"))
            if query is None:
                tasks = {
                    "RA": "Assess the cardiac risk using the supplied clinical history.",
                    "QA": "Interpret the supplied history and answer its clinical questions.",
                    "keywords": "Extract the medical entities from the supplied clinical history.",
                    "Conversation": "Continue and correct the clinical dialogue using the history.",
                }
                for key, instruction in tasks.items():
                    if key in row:
                        questions = _source_questions(row[key]) if key == "QA" else []
                        query = "\n".join(questions) if questions else instruction
                        reference = row[key]
                        break
            if query is None or reference is None:
                raise ValueError(f"Unrecognized QA schema in {path}:{index}")
            yield {
                "dataset": "expert-qa",
                "record_id": f"{path.stem}:{index}",
                "task": path.stem,
                "patient_id": str(row["subject_id"]) if "subject_id" in row else None,
                "study_id": str(row["study_id"]) if "study_id" in row else None,
                "query": str(query),
                "reference_answer": reference,
                "context": row.get("result", row.get("medical_advice", "")),
                "field_policy": "A=query,Q=answer" if cross_modal else "source task fields",
                "raw": row,
            }


def index_dataset(
    dataset: str,
    root: Path,
    output: Path,
    *,
    sampling_rate: int = 500,
    check_waveforms: bool = True,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        iter_expert_qa(root)
        if dataset == "expert-qa"
        else iter_waveform_metadata(dataset, root, sampling_rate=sampling_rate)
    )
    patients: set[str] = set()
    tasks: Counter[str] = Counter()
    folds: Counter[str] = Counter()
    missing_waveforms = 0
    records = 0
    temporary = output.with_name(output.name + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            records += 1
            if row.get("patient_id"):
                patients.add(row["patient_id"])
            if "task" in row:
                tasks[row["task"]] += 1
            if "fold" in row:
                folds[str(row["fold"])] += 1
            if check_waveforms and "waveform_path" in row:
                base = root / row["waveform_path"]
                if not all(Path(str(base) + suffix).is_file() for suffix in (".hea", ".dat")):
                    missing_waveforms += 1
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if not records:
        raise ValueError(f"No source records found for {dataset}")
    temporary.replace(output)
    spec = PAPER_DATASETS[dataset]
    receipt_path = root / "download_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    return {
        "dataset": dataset,
        "source": spec["source"],
        "version": receipt.get("version"),
        "requested_version": spec["version"],
        "root": str(root.resolve()),
        "index": str(output.resolve()),
        "index_sha256": sha256_file(output),
        "records": records,
        "identified_patients": len(patients),
        "tasks": dict(tasks),
        "folds": dict(folds),
        "paper_records": spec["paper_records"],
        "record_count_matches_paper": records == spec["paper_records"],
        "waveforms_checked": check_waveforms and dataset != "expert-qa",
        "missing_waveforms": missing_waveforms,
        "all_source_rows_indexed": True,
        "complete": missing_waveforms == 0 if check_waveforms else False,
    }


def prepare_paper_data(args) -> int:
    with _acquisition_lock(Path(args.output) / ".prepare.lock"):
        return _prepare_paper_data(args)


def _prepare_paper_data(args) -> int:
    data_root = Path(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = list(dict.fromkeys(args.datasets))
    manifest = {
        "schema_version": "care-ecg-paper-data.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
        "pending": selected.copy(),
        "errors": {},
        "patient_or_record_caps": None,
        "complete": False,
        "protocol": "Full-corpus inference indexes; never merge training and held-out folds.",
        "limitations": [
            "The paper does not release its exact cohorts or training/QA splits.",
            "Released QA task rows differ from the paper's 47,211 QA-pair count.",
            "QA records without patient IDs are retained without invented patient identities.",
        ],
    }

    def save_manifest():
        temporary = output / "manifest.json.part"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    def prepare_one(dataset):
        override = getattr(args, dataset.replace("-", "_") + "_root", None)
        root = Path(override) if override else data_root / str(PAPER_DATASETS[dataset]["directory"])
        if args.download and override is None:
            root = acquire_dataset(dataset, data_root)
        return index_dataset(
            dataset,
            root,
            output / f"{dataset}.jsonl",
            sampling_rate=args.sampling_rate,
            check_waveforms=True,
        )

    save_manifest()
    with ThreadPoolExecutor(max_workers=len(selected) or 1) as executor:
        futures = {executor.submit(prepare_one, dataset): dataset for dataset in selected}
        for future in as_completed(futures):
            dataset = futures[future]
            manifest["pending"].remove(dataset)
            try:
                summary = future.result()
            except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
                manifest["errors"][dataset] = str(exc)
                print(f"{dataset}: {exc}", file=sys.stderr, flush=True)
            else:
                manifest["datasets"].append(summary)
                print(json.dumps(summary, indent=2), flush=True)
            manifest["complete"] = (
                not manifest["pending"]
                and not manifest["errors"]
                and all(item["complete"] for item in manifest["datasets"])
            )
            save_manifest()
    return 0 if manifest["complete"] else 1
