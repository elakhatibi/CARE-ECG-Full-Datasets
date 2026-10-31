CARE-ECG: Detailed Methodology, Experimental Protocols, and Evidence-Based Results
Evidence review: 28 October 2026

Open CARE_ECG_Detailed_Methodology_and_Results.pdf (56 pages).
The report contains 17 main sections, 9 appendices, 11 figures and 24 tables.

Contents
  manuscript.tex              Architecture, datasets and core mathematical methods
  experiments_results.tex     Classification, historical security and interpretation
  adaptive_dpo.tex             Adaptive v2, finite-action DPO, co-evolution and results
  appendices.tex              Supplementary results, exact commands and references
  build_evidence.py           Classification/security aggregate evidence builder
  build_extensions.py         Adaptive/DPO evidence, loss reconstruction and figures
  validate_report.py          PDF, scope, layout and package integrity checks
  evidence.json               Aggregate outcomes and input/source SHA-256 digests
  software_verification.json  Current software verification and test-count evidence
  report_validation.json      Report-specific check results and PDF digest
  tables/                     Machine-readable CSV and generated TeX tables
  figures/                    Vector PDF figures and PNG versions
  SHA256SUMS.txt              Digests of every file in the shareable package

Coverage
The manuscript explains preprocessing; InceptionTime/VAE representation learning;
ordered K2/BDeu Bayesian reasoning; latent state edits; diagnostic stacking;
history, retrieval, response and verification agents; full-corpus ingestion;
waveform threat models; constraints and query accounting; adaptive v2 optimization;
two frozen-reference DPO controllers; preference construction; patient weighting;
validation promotion, coverage guards and rollback; and deployment/hash contracts.

Results include the complete eligible PTB-XL fixed-candidate fold-10 predictions,
patient-cluster uncertainty, class errors, calibration, specialist screening,
lexical report checks, historical validation/held-out security matrices, the
nine-attack v2 validation comparison, and two real DPO development rounds.
The DPO candidates reduced training loss but neither role passed promotion.
Unrun full-cohort, held-out DPO, MIMIC and QA evaluations are explicitly marked.

Rebuild from the repository root, with the referenced local artifacts available:
  /usr/bin/python3 reports/methodology-20261028/build_evidence.py

Then from reports/methodology-20261028:
  pdflatex -interaction=nonstopmode -halt-on-error \
    -jobname=CARE_ECG_Detailed_Methodology_and_Results manuscript.tex
  pdflatex -interaction=nonstopmode -halt-on-error \
    -jobname=CARE_ECG_Detailed_Methodology_and_Results manuscript.tex
  /usr/bin/python3 validate_report.py

Repeat TeX if it requests another reference pass. The evidence builder uses the
system scientific Python installation (NumPy 1.26.4, Matplotlib 3.6.3), without
changing the project's machine-learning environment. Retrospective classification
bootstrap settings remain 10,000 resamples and seed 20261027; the adaptive-v2
paired analysis uses 10,000 resamples and seed 42. DPO losses are reconstructed
from the actual saved preference records and frozen checkpoints, not retrained.

The authoring sources, rendered figures and aggregate tables can be edited and
compiled from the extracted package alone. Rebuilding numerical evidence also
requires the separate raw datasets, model artifacts and experiment outputs at
their original repository paths. Those are not included in this shareable ZIP.

The package includes aggregate outcomes, hashes and local source paths only.
It contains no patient identifiers, waveform arrays, credentials or model weights.
Historical experiment code identities are distinct from current source hashes.
The earlier 27 October report is preserved in its own directory. Publication of
the earlier repository snapshot does not mean these local updates are uploaded.

A later rebuild can change data-status snapshots or source identities. Review
the prose against the rebuilt evidence before treating it as a new report.
