# EchoSynth

Reference implementation for *Synthetic Echocardiograms from Diffusion Models in Rare
Cardiovascular Disease*. The public pipeline is manifest-driven and deterministic. It never
downloads data or weights at runtime.

## Revision-analysis status

This reference pipeline implements the revised evaluation unit: it assigns every encounter from
the same patient to one partition, averages image predictions within encounter and then averages
encounter predictions within patient, and evaluates held-out patients. Patient identifiers are
hashed before manifest export. Protected manifests, images, model weights, and institutional run
artifacts are not included in this repository, so the reported clinical results cannot be
reconstructed without the governed Duke data.

## Which analysis does this release implement?

| Setting | Original primary Duke analysis | Full-archive patient sensitivity |
|---|---|---|
| Split | Encounter (repeat-patient overlap not excluded) | Unique patient, 70/15/15 |
| Reported unit | Encounter | Patient; encounter outputs are secondary |
| Reverse DDIM steps | 500 | 100 |
| Classifier seeds | One configuration fit | 2024, 2025, 2026, 2027, 2028 |
| Synthetic-count grid | 0–2000, step 100 | 500, 1000, 1500, 2000 |
| Classifier batch size | 32 in the reference specification | 64 in the full-archive configuration |
| Diffusion optimization steps | Original run provenance not reconstructed here | 10000, explicitly supplied on CLI |

The root package is a general public **reference implementation**, not a bit-for-bit
archive of either institutional run. Its current classifier aggregates by patient even
though legacy YAML contains an encounter-unit field. Do not use that field to claim
reconstruction of the original encounter-level results. The original predictions and
selected synthetic count are not distributed or reconstructed by this release.

For the full-archive workflow, use `workflows/patient_sensitivity/` and its separate
source tree/configurations; see [the runbook](workflows/patient_sensitivity/README.md).
This source snapshot is organized from the local revision workflow, not certified as
identical to every script executed on PRCC. It records the distinction rather than
silently substituting a different pipeline. Protected institutional data and run
artifacts are required for clinical result reproduction.

## Installation

Use Python 3.11 in an isolated environment:

```bash
conda env create -f environment.yml
conda activate echosynth
pre-commit install
make test
```

The legacy Keras view model is deliberately isolated. Create a second environment from
`requirements-view.txt`, or use the preferred ONNX inference path. Never import TensorFlow
and PyTorch in the same process.

## Generic reference pipeline (not the archived clinical run)

1. Ingest DICOM or EchoNet data into a validated manifest.
2. Run the eight-feature QC screen and manually review flagged frames.
3. Register, sector-mask, resize, and crop frames to 112×112 grayscale.
4. Optionally classify views; the primary endpoint remains all-view.
5. Create unique-patient train/validation/test splits, keeping every encounter from one patient together.
6. Train a class-conditional diffusion model on the training partition only.
7. Sample synthetic training images using explicitly configured DDIM steps (legacy default 500).
8. Train CompactCNN classifiers and select checkpoints on patient-level validation AUROC.
9. Select the synthetic count on validation data, then access test exactly once.
10. Evaluate patient-level AUROC/AUPRC and paired DeLong intervals after two-stage aggregation.
11. Run fidelity, discriminator, embedding, and descriptive survey analyses.
12. Regenerate tables and figures solely from run artifacts.

Run `make <target>` for `ingest`, `qc`, `preprocess`, `views`, `splits`, `gen`, `sample`,
`train`, `experiment`, `fidelity`, or `report`. Every command accepts local paths; there is no
runtime network dependency.

### Generic reference example (not the manuscript sensitivity runbook)

Mount de-identified source data outside this repository, then run the following in order:

```bash
python scripts/00_ingest_dicom.py --root /mounted/duke --out-dir /derived/frames \
  --manifest manifests/duke_master.csv --config configs/data/duke.yaml
python scripts/01_run_qc.py --manifest manifests/duke_master.csv --config configs/data/duke.yaml
# Complete manifests/qc_review_queue.csv, then apply the reviewed copy:
python scripts/01_run_qc.py --manifest manifests/duke_master.csv --config configs/data/duke.yaml \
  --apply-review /reviewed/qc_reviewed.csv
python scripts/04_make_splits.py --manifest manifests/duke_master.csv --seed 2024
python scripts/02_preprocess.py --manifest manifests/duke_train.csv --out-dir /derived/112 \
  --config configs/data/duke.yaml --build-reference
python scripts/02_preprocess.py --manifest manifests/duke_master.csv --out-dir /derived/112 \
  --config configs/data/duke.yaml
# Optional; the primary endpoint is all-view and does not require this command:
python scripts/03_classify_views.py --manifest manifests/duke_master.csv \
  --weights /weights/echo-view.onnx --mode onnx
python scripts/05_train_diffusion.py --config configs/diffusion/duke_ca.yaml \
  --manifest manifests/duke_train.csv --out-dir results/runs/diffusion/ckpt
python scripts/06_sample_synthetic.py --config configs/diffusion/duke_ca.yaml \
  --ckpt results/runs/diffusion/ckpt/ema_best.pt --counts '{CA: 3000}' \
  --out-dir results/synthetic --manifest-out manifests/synthetic_duke_ca.csv
python scripts/08_run_experiment.py --config configs/experiment/duke_two_step_allview.yaml
python scripts/09_fidelity.py --real-manifest manifests/duke_master.csv \
  --synthetic-manifest manifests/synthetic_duke_ca.csv --output results/fidelity.json
python scripts/12_make_report.py --runs results/runs --out-dir results/report
```

Real data, DICOMs, derived images, model weights, manifests, and checkpoints are gitignored.
The view weights are obtained separately from the
[MIT reference project](https://github.com/raventan95/echo-view-classifier); missing weights
produce a download link rather than a traceback. The test-access audit file in each run must
contain exactly one event.

## Where the numbers come from

| Reported output | Source of truth | Generated by |
|---|---|---|
| Duke and EchoNet AUROC targets | `PROJECT_SPEC.md` §7 | reproduction tests only |
| Fidelity targets | `PROJECT_SPEC.md` §7 | reproduction tests only |
| Survey targets | `PROJECT_SPEC.md` §7 | reproduction tests only |
| Actual run metrics | `results/runs/*/metrics.json` | experiment/evaluation code |
| Tables and figures | run artifacts on disk | `make report` |

Target values are legacy range checks for a mounted metrics artifact, never literals returned by analysis code. Passing these checks is not independent reproduction of the paper. The withdrawn EchoNet-LVH paired interval is excluded.
See `docs/DECISIONS.md` for manuscript ambiguities and their recorded resolutions.
