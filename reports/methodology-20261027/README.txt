CARE-ECG: Detailed Methodology, Experimental Protocols, and Evidence-Based Results
Report date: 27 October 2026

Open CARE_ECG_Detailed_Methodology_and_Results.pdf (43 pages).

Contents
  manuscript.tex              Main methodology and mathematical definitions
  experiments_results.tex     Experimental design, results, and interpretation
  appendices.tex              Complete supplementary tables, replay, references
  build_evidence.py           Aggregate reanalysis and scientific figure builder
  evidence.json               Aggregate results and input/source SHA-256 digests
  software_verification.json  Current successful test-run evidence
  tables/                     Machine-readable CSV and generated TeX tables
  figures/                    Vector PDF figures and PNG previews

Measured results come from existing local experiment artifacts. The report adds
retrospective patient-cluster bootstrap uncertainty; it does not train or tune
models. Data acquisition and evaluation completeness are explicitly distinguished.
No patient identifiers or waveform arrays are included in this report package.
Raw datasets, model weights, individual predictions, and archived workbooks remain
separate local artifacts and are required to regenerate all numerical evidence.

Rebuild from the repository root (scientific Python with NumPy and Matplotlib):
  /usr/bin/python3 reports/methodology-20261027/build_evidence.py

Then from reports/methodology-20261027:
  pdflatex -interaction=nonstopmode -halt-on-error \
    -jobname=CARE_ECG_Detailed_Methodology_and_Results manuscript.tex
  pdflatex -interaction=nonstopmode -halt-on-error \
    -jobname=CARE_ECG_Detailed_Methodology_and_Results manuscript.tex

The local system Python plotting environment was used without changing the ML
environment: Python 3.12, NumPy 1.26.4, Matplotlib 3.6.3. The generator checks saved
metrics, record/label/probability alignment, patient separation, selected provenance
hashes, and security point estimates against observation-level results.

Important interpretation
  Released stack classification and internal waveform-victim security results use
  different probability sources. Clinical incorrect release is not evaluable.
  PTB-XL 1.0.1 and released Expert-ECG-QA indexing are complete at the snapshot;
  MIMIC acquisition and full benchmark evaluations are not all complete.
  Original-paper exact reproduction is not claimed.

All authoring sources are editable. A later evidence rebuild can change dataset
status or source hashes; review the prose before publishing an updated snapshot.
