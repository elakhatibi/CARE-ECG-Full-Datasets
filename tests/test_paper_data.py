import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from care_ecg.cli import _limit_eligible_cases, _security_cohort_cases, build_parser
from care_ecg.config import CAREConfig
from care_ecg.data import ECGArrayBundle
from care_ecg.paper_data import (
    download_file,
    extract_archive,
    index_dataset,
    iter_expert_qa,
    iter_waveform_metadata,
    prepare_paper_data,
    verify_checksums,
)
from care_ecg.paper_inference import answer_qa_index, load_waveform, predict_waveform_index


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PaperDataTests(unittest.TestCase):
    def test_full_ptbxl_index_retains_unlabeled_and_repeated_patient_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "ecg_id": "1",
                    "patient_id": "p1",
                    "scp_codes": "{'NORM': 100}",
                    "strat_fold": "1",
                    "filename_lr": "one",
                    "filename_hr": "one_hr",
                },
                {
                    "ecg_id": "2",
                    "patient_id": "p1",
                    "scp_codes": "{'SR': 100}",
                    "strat_fold": "1",
                    "filename_lr": "two",
                    "filename_hr": "two_hr",
                },
                {
                    "ecg_id": "3",
                    "patient_id": "p2",
                    "scp_codes": "{}",
                    "strat_fold": "10",
                    "filename_lr": "three",
                    "filename_hr": "three_hr",
                },
            ]
            write_csv(root / "ptbxl_database.csv", rows[0].keys(), rows)
            for name in ("one", "two", "three"):
                for suffix in (".hea", ".dat"):
                    (root / (name + suffix)).touch()
            output = root / "all.jsonl"
            summary = index_dataset("ptbxl", root, output, sampling_rate=100)
            indexed = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(summary["records"], 3)
            self.assertEqual(summary["identified_patients"], 2)
            self.assertTrue(summary["complete"])
            self.assertEqual(indexed[1]["scp_codes"], {"SR": 100})
            self.assertEqual(indexed[2]["scp_codes"], {})
            self.assertEqual([r["fold"] for r in indexed], [1, 1, 10])
            # A local override cannot silently claim to be the paper's release.
            self.assertIsNone(summary["version"])

    def test_mimic_keeps_all_studies_without_synthetic_class_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"subject_id": "p1", "study_id": "s1", "ecg_time": "2100-01-01", "path": "files/a"},
                {"subject_id": "p1", "study_id": "s2", "ecg_time": "2100-01-02", "path": "files/b"},
            ]
            write_csv(root / "record_list.csv", rows[0].keys(), rows)
            loaded = list(iter_waveform_metadata("mimic", root))
            self.assertEqual([r["record_id"] for r in loaded], ["s1", "s2"])
            self.assertTrue(all("label" not in row for row in loaded))
            summary = index_dataset("mimic", root, root / "all.jsonl")
            self.assertEqual(summary["missing_waveforms"], 2)
            self.assertFalse(summary["complete"])

    def test_qa_formats_keep_all_rows_nested_targets_and_reversed_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "ECG Knowledge.json": [{"Q": "What is ECG?", "A": "Electrical recording."}],
                "Cross modal diagnosis (ECG Text corresponding diagnosis).json": [
                    {
                        "A": "Interpret ECG",
                        "Q": "REFERENCE ONLY",
                        "subject_id": 123,
                        "study_id": 456,
                    }
                ],
                "Memory correction.json": [
                    {
                        "subject_id": 123,
                        "result": "History",
                        "Conversation": "{'nested': ['dialogue']}",
                    }
                ],
                "Patient rights.json": [{"Question": "Can I ask?", "Answer": "Yes."}],
            }
            for name, rows in fixtures.items():
                (root / name).write_text(json.dumps(rows))
            rows = list(iter_expert_qa(root))
            self.assertEqual(len(rows), 4)
            cross = next(row for row in rows if row["study_id"])
            self.assertEqual(cross["query"], "Interpret ECG")
            self.assertEqual(cross["reference_answer"], "REFERENCE ONLY")
            self.assertEqual(cross["study_id"], "456")
            self.assertEqual(sum(row["patient_id"] is None for row in rows), 2)

    def test_nested_qa_extracts_source_questions_without_reference_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "clinical task": {
                    "Question": "Actual source question?",
                    "Answer": "SECRET_REFERENCE",
                }
            }
            (root / "Long text diagnosis.json").write_text(
                json.dumps([{"subject_id": 1, "result": "Input history", "QA": repr(payload)}])
            )
            row = next(iter_expert_qa(root))
            self.assertEqual(row["query"], "Actual source question?")
            self.assertNotIn("SECRET_REFERENCE", row["query"])
            self.assertIn("SECRET_REFERENCE", row["reference_answer"])

    def test_archive_extraction_and_publisher_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "release.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("release/a.dat", b"waveform")
                digest = hashlib.sha256(b"waveform").hexdigest()
                handle.writestr("release/SHA256SUMS.txt", digest + "  a.dat\n")
            extract_archive(archive, root / "data")
            self.assertEqual(verify_checksums(root / "data")["verified_files"], 1)
            (root / "data/a.dat").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                verify_checksums(root / "data")
            with zipfile.ZipFile(root / "bad.zip", "w") as handle:
                handle.writestr("../outside", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_archive(root / "bad.zip", root / "bad")

    def test_http_resume_restarts_when_server_ignores_range(self):
        from io import BytesIO

        class Response(BytesIO):
            status = 200

            def __init__(self, content):
                super().__init__(content)
                self.headers = {"Content-Length": "8"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.zip"
            output.with_name(output.name + ".part").write_bytes(b"partial")
            with patch("urllib.request.urlopen", return_value=Response(b"complete")) as opener:
                download_file("https://example.test/release.zip", output)
            self.assertEqual(output.read_bytes(), b"complete")
            self.assertEqual(opener.call_args.args[0].headers["Range"], "bytes=7-")

    def test_http_range_resume_accepts_standard_content_range_header(self):
        from io import BytesIO

        class Response(BytesIO):
            status = 206

            def __init__(self):
                super().__init__(b"def")
                self.headers = {"Content-Length": "3", "Content-Range": "bytes 3-5/6"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.zip"
            output.with_name(output.name + ".part").write_bytes(b"abc")
            with patch("urllib.request.urlopen", return_value=Response()):
                download_file("https://example.test/release.zip", output)
            self.assertEqual(output.read_bytes(), b"abcdef")

    def test_http_resume_rejects_wrong_range_without_appending_corrupt_bytes(self):
        from io import BytesIO

        class Response(BytesIO):
            status = 206

            def __init__(self):
                super().__init__(b"xyz")
                self.headers = {"Content-Length": "3", "Content-Range": "bytes 0-2/6"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.zip"
            partial = output.with_name(output.name + ".part")
            partial.write_bytes(b"abc")
            with (
                patch("urllib.request.urlopen", return_value=Response()),
                self.assertRaisesRegex(ValueError, "mismatched download range"),
            ):
                download_file("https://example.test/release.zip", output, retries=1)
            self.assertEqual(partial.read_bytes(), b"abc")
            self.assertFalse(output.exists())

    def test_http_timeouts_with_progress_do_not_exhaust_retry_budget(self):
        from io import BytesIO

        class Response(BytesIO):
            def __init__(self, content, offset, interrupted):
                super().__init__(content)
                self.status = 206 if offset else 200
                self.headers = {"Content-Length": str(9 - offset)}
                if offset:
                    self.headers["Content-Range"] = f"bytes {offset}-8/9"
                self.interrupted = interrupted

            def read(self, size=-1):
                chunk = super().read(size)
                if not chunk and self.interrupted:
                    raise TimeoutError("simulated transient read timeout")
                return chunk

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.zip"
            with (
                patch(
                    "urllib.request.urlopen",
                    side_effect=[
                        Response(b"abc", 0, True),
                        Response(b"def", 3, True),
                        Response(b"ghi", 6, False),
                    ],
                ) as opener,
                patch("care_ecg.paper_data.time.sleep"),
            ):
                download_file("https://example.test/release.zip", output, retries=1)
            self.assertEqual(output.read_bytes(), b"abcdefghi")
            self.assertEqual(opener.call_count, 3)
            self.assertEqual(opener.call_args_list[1].args[0].headers["Range"], "bytes=3-")
            self.assertEqual(opener.call_args_list[2].args[0].headers["Range"], "bytes=6-")

    def test_http_stops_after_consecutive_failures_without_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.zip"
            with (
                patch("urllib.request.urlopen", side_effect=TimeoutError("no progress")) as opener,
                patch("care_ecg.paper_data.time.sleep"),
                self.assertRaises(TimeoutError),
            ):
                download_file("https://example.test/release.zip", output, retries=3)
            self.assertEqual(opener.call_count, 3)
            self.assertFalse(output.exists())

    def test_prepare_reports_missing_dataset_and_keeps_other_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa = root / "ecg-expert-qa"
            qa.mkdir()
            (qa / "Basic.json").write_text('[{"Q":"q","A":"a"}]')
            args = SimpleNamespace(
                data_root=root,
                output=root / "indexes",
                datasets=["mimic", "expert-qa"],
                download=False,
                sampling_rate=500,
            )
            self.assertEqual(prepare_paper_data(args), 1)
            manifest = json.loads((root / "indexes/manifest.json").read_text())
            self.assertIn("mimic", manifest["errors"])
            self.assertEqual(manifest["datasets"][0]["records"], 1)
            self.assertFalse(manifest["complete"])
            self.assertFalse(manifest["pending"])

    def test_waveform_loader_reorders_leads_converts_units_and_uses_header_rate(self):
        import sys

        from care_ecg.data import LEAD_ORDER

        values = np.tile(np.arange(12, dtype=np.float32), (4, 1))
        fields = {"sig_name": list(reversed(LEAD_ORDER)), "units": ["uV"] * 12, "fs": 2}
        fake = SimpleNamespace(rdsamp=lambda _: (values, fields))
        config = CAREConfig()
        config.data.sample_rate = 2
        config.data.seconds = 2
        config.data.normalization = "none"
        with patch.dict(sys.modules, {"wfdb": fake}):
            loaded = load_waveform({"waveform_path": "a", "sampling_rate": 500}, Path("."), config)
        np.testing.assert_allclose(loaded[:, 0], np.arange(11, -1, -1) * 0.001)
        self.assertEqual(loaded.shape, (12, 4))

    def test_streaming_inference_batches_all_records_and_preserves_patient_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.jsonl"
            rows = [
                {
                    "dataset": "mimic",
                    "record_id": str(i),
                    "patient_id": "p1",
                    "waveform_path": str(i),
                }
                for i in range(5)
            ]
            index.write_text("".join(json.dumps(row) + "\n" for row in rows))
            pipeline = SimpleNamespace(
                config=CAREConfig(),
                encode_batch_with_neural=lambda signals, batch_size: (
                    np.zeros((len(signals), 2)),
                    None,
                ),
                run_encoded=lambda *args, **kwargs: SimpleNamespace(
                    to_dict=lambda: {"metadata": {}, "diagnosis": "NORM"}
                ),
            )
            args = SimpleNamespace(
                artifacts="unused",
                index=index,
                root=root,
                output=root / "predictions.jsonl",
                batch_size=2,
                folds=None,
                query="question",
            )
            with (
                patch("care_ecg.paper_inference._load_pipeline", return_value=pipeline),
                patch("care_ecg.paper_inference.load_waveform", return_value=np.zeros((12, 4))),
                patch.object(
                    pipeline, "encode_batch_with_neural", wraps=pipeline.encode_batch_with_neural
                ) as encode,
            ):
                self.assertEqual(predict_waveform_index(args), 0)
            self.assertEqual([len(c.args[0]) for c in encode.call_args_list], [2, 2, 1])
            predictions = [json.loads(line) for line in args.output.read_text().splitlines()]
            self.assertEqual(len(predictions), 5)
            self.assertTrue(all(row["patient_id"] == "p1" for row in predictions))
            self.assertTrue(all("expected" not in row for row in predictions))

    def test_qa_reference_answers_are_never_in_generation_prompt(self):
        prompts = []

        def generate(prompt, facts):
            prompts.append(prompt)
            return "Insufficient retrieved evidence."

        pipeline = SimpleNamespace(
            config=CAREConfig(),
            retriever=SimpleNamespace(retrieve=lambda *args: []),
            response_agent=SimpleNamespace(
                generator=SimpleNamespace(generate=generate),
                fuzzy_match_threshold=0.32,
                fallback_to_rag_only=True,
                hallucination_threshold=0.5,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "qa.jsonl"
            row = {
                "query": "q",
                "record_id": "r1",
                "patient_id": None,
                "task": "Basic",
                "reference_answer": "SECRET_REFERENCE",
                "raw": {},
                "context": "Input",
            }
            index.write_text(json.dumps(row) + "\n")
            args = SimpleNamespace(
                index=index, output=root / "answers.jsonl", mimic_index=None, mimic_root=None
            )
            with patch("care_ecg.paper_inference._load_pipeline", return_value=pipeline):
                self.assertEqual(answer_qa_index(args), 0)
            self.assertNotIn("SECRET_REFERENCE", prompts[0])
            result = json.loads(args.output.read_text())
            self.assertEqual(result["reference_answer"], "SECRET_REFERENCE")
            self.assertEqual(result["input_mode"], "text_only")

    def test_uncapped_security_keeps_repeated_records_and_caps_count_patients(self):
        signals = np.zeros((4, 1, 4), dtype=np.float32)
        bundle = ECGArrayBundle(
            signals=signals,
            labels=np.zeros(4, dtype=int),
            patient_ids=np.asarray(["p1", "p2", "p1", "p3"]),
            record_ids=np.asarray(["r1", "r2", "r3", "r4"]),
        )
        cases = _security_cohort_cases(bundle, signals, ("MI",), ("T1",), all_records=True)
        self.assertEqual(len(cases), 4)
        eligible = [SimpleNamespace(case=case) for case in cases]
        self.assertEqual(len(_limit_eligible_cases(eligible, None)), 4)
        self.assertEqual(
            [x.case.record_id for x in _limit_eligible_cases(eligible, 1)], ["r1", "r3"]
        )
        capped = _security_cohort_cases(
            bundle,
            signals,
            ("MI",),
            ("T1",),
            all_records=True,
            max_candidates_per_target=1,
        )
        self.assertEqual([case.record_id for case in capped], ["r1", "r3"])
        args = build_parser().parse_args(
            ["security-run", "a", "d", "o", "--all-patients", "--records-per-patient", "all"]
        )
        self.assertTrue(args.all_patients)


if __name__ == "__main__":
    unittest.main()
