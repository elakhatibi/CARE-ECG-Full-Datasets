"""Validate and package the completed aggregate-only report.

Run after the evidence builder and a reference-stable LaTeX build.
Uses Python's standard library plus Poppler command-line utilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "CARE_ECG_Detailed_Methodology_and_Results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pdf = HERE / (STEM + ".pdf")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
    assert pages == 56, ("Review changed pagination and update README", pages)
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    (HERE / (STEM + ".txt")).write_text(text)
    bbox = subprocess.check_output(["pdftotext", "-bbox", str(pdf), "-"], text=True)
    # Poppler can emit legacy math-font control glyphs in XML text nodes.
    # Remove invalid XML controls before checking the actual word coordinates.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", bbox)
    document = ET.fromstring(cleaned)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    bounds_errors, words = [], 0
    for i, page in enumerate(document.findall(".//x:page", ns), 1):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.findall(".//x:word", ns):
            words += 1
            values = [float(word.attrib[k]) for k in ("xMin", "yMin", "xMax", "yMax")]
            if not all(math.isfinite(v) for v in values) or not (
                -1 <= values[0] <= values[2] <= width + 1
                and -1 <= values[1] <= values[3] <= height + 1
            ):
                bounds_errors.append({"page": i, "word": word.text, "bounds": values})
    assert not bounds_errors, bounds_errors
    log = (HERE / "latex-final.log").read_text()
    assert not re.search(r"Overfull|undefined|Rerun|LaTeX Error|^!", log, re.MULTILINE)
    aux = (HERE / (STEM + ".aux")).read_text()
    figures = aux.count(r"\contentsline {figure}")
    tables = aux.count(r"\contentsline {table}")
    assert figures == 11 and tables == 24
    evidence = json.loads((HERE / "evidence.json").read_text())
    assert evidence["adaptive_v2"]["all_case_keys_match"]
    assert all(c["loss_matches"] for c in evidence["dpo"]["loss_reconstruction_checks"])
    assert all(b["verified"] for b in evidence["dpo"]["runtime_policy_bindings"])
    assert all(
        not r[role + "_promoted"]
        for r in evidence["dpo"]["rounds"]
        for role in ("attacker", "verifier")
    )
    assert evidence["dataset_manifest"]["pending"] == ["mimic"]
    assert "not evaluable" in text and "no candidate is promoted" in text
    forbidden_keys = {"patient_id", "patient_ids", "record_id", "record_ids"}

    def check_aggregate(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value), (
                "Patient-level data in shareable evidence"
            )
            for child in value.values():
                check_aggregate(child)
        elif isinstance(value, list):
            for child in value:
                check_aggregate(child)

    check_aggregate(evidence)
    validation = {
        "report_date": "2026-10-28",
        "pdf_sha256": digest(pdf),
        "pdf_bytes": pdf.stat().st_size,
        "pages": pages,
        "figures": figures,
        "tables": tables,
        "extracted_words": words,
        "compiler": "pdflatex",
        "unresolved_references": False,
        "overfull_boxes": False,
        "word_bounds_errors": bounds_errors,
        "aggregate_only_evidence_checked": True,
        "adaptive_case_alignment_checked": True,
        "four_dpo_losses_reconstructed": True,
        "zero_promotions_verified": True,
        "runtime_policy_lock_bindings_checked": True,
        "visual_review": "All pages reviewed in contact sheets; adaptive results, DPO math, loop diagram, curves and tables also reviewed individually.",
    }
    (HERE / "report_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    names = [
        "manuscript.tex",
        "experiments_results.tex",
        "adaptive_dpo.tex",
        "appendices.tex",
        "build_evidence.py",
        "build_extensions.py",
        "validate_report.py",
        "README.txt",
        "evidence.json",
        "software_verification.json",
        "report_validation.json",
        "pytest.log",
        "pytest_collection.log",
        "latex-final.log",
        STEM + ".pdf",
        STEM + ".txt",
    ]
    paths = [HERE / name for name in names]
    paths += sorted((HERE / "figures").glob("*.pdf"))
    paths += sorted((HERE / "figures").glob("*.png"))
    paths += sorted((HERE / "tables").glob("*.csv"))
    paths += sorted((HERE / "tables").glob("*.tex"))
    sums = HERE / "SHA256SUMS.txt"
    sums.write_text("".join(f"{digest(p)}  {p.relative_to(HERE)}\n" for p in paths))
    paths.append(sums)
    archive = HERE / "CARE_ECG_Report_and_Sources.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(HERE))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path in paths:
            assert hashlib.sha256(z.read(str(path.relative_to(HERE)))).hexdigest() == digest(path)
    print(
        json.dumps(
            {
                **validation,
                "archive_files": len(paths),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": digest(archive),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
