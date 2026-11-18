# Full-archive patient-level sensitivity workflow

This separate source tree organizes the locally retained revision workflow and its
post-processing scripts. It is not the original encounter-level analysis and is not
certified as a byte-identical snapshot of every PRCC-executed file. Do not infer
clinical reproduction from mock tests or from legacy target-range assertions.

## Configuration and runtime

Use Python 3.12 and the accompanying `requirements.txt` (PyTorch 2.4.1 in the
institutional environment). Run from this directory with its `src` first on
`PYTHONPATH`; do not accidentally import the generic root `echosynth` package:

```bash
cd workflows/patient_sensitivity
export PYTHONPATH="$PWD/src"
python -c 'import echosynth; print(echosynth.__file__)'
```

Mount authorized data and preserve the relative image paths in the manifests.
This release contains no patient manifests, raw identifiers, hashing salt,
DICOMs, generated images, model weights, or institutional outputs. Preparation
requires a private site configuration (source groups and private hashing salt);
it is deliberately not supplied. The automated source preparation and the generic
root QC module are separate code paths; the presence of QC columns is not proof
that eight-feature QC was applied.

For previously prepared authorized manifests, the main entry points are:

```bash
python scripts/05_train_diffusion.py \
  --config configs/diffusion/ca_revision_prcc_full.yaml \
  --manifest work/full_revision/manifests/ca_train.csv \
  --out-dir results/full_revision/diffusion/ckpt --steps 10000
python scripts/06_sample_synthetic.py \
  --config configs/diffusion/ca_revision_prcc_full.yaml \
  --ckpt results/full_revision/diffusion/ckpt/ema_best.pt \
  --counts '{CA: 2000, LVH: 2000, GEN: 2000}' \
  --out-dir results/full_revision/synthetic \
  --manifest-out work/full_revision/manifests/ca_synthetic.csv
for SEED in 2024 2025 2026 2027 2028; do
  python scripts/08_run_experiment.py \
    --config configs/experiment/ca_revision_prcc_full.yaml \
    --set seed=$SEED --set run_id=ca_revision_full_seed$SEED
done
```

The generator uses 1000 forward diffusion steps, 100 reverse DDIM steps, guidance
2.0, seed 2024, and **10000 optimization steps explicitly passed on the CLI**.
The classifier configuration specifies batch 64, maximum 100 epochs, patience 15,
Adam LR 0.001, and a 500/1000/1500/2000 synthetic-count grid. The shared generator
is trained once; classifier seeds are not independent generator repetitions.
Synthetic CA images enter training only. The source supports additional comparator
arms; sampling 6000 images does not mean all 6000 were used in CA augmentation.

## Supplemental analysis entry points

All outputs require actual run artifacts. Do not use mock data as manuscript results.
Set `V6` to the prepared project root (absolute path) and `FOLLOW` to an output root.

```bash
python analyses/followup/01_calibration.py --v6-root "$V6" \
  --out-dir "$FOLLOW/calibration" --bootstrap 2000
python analyses/followup/03_view_phase_audit.py --help
python analyses/followup/04_privacy_nearest_neighbor.py --v6-root "$V6" \
  --out-dir "$FOLLOW/privacy" --max-images 1500 --bootstrap 2000
python analyses/followup/05_run_no_flip.py --v6-root "$V6" \
  --follow-root "$FOLLOW" --seed 2024
python analyses/operating_point/final_operating_point.py \
  --results-root "$V6/results/full_revision"
```

For acquisition analysis, copy `analyses/acquisition/config.yaml` to a private
location, set `project_root` to the authorized root, and run:

```bash
python analyses/acquisition/era_vendor_analysis.py --config /private/acquisition.yaml
```

That script writes into `analyses/acquisition/results` in this source checkout,
including a **patient-level** linkage output. Keep those outputs inside the
protected environment; only aggregate reviewed results belong in a paper.
The included configuration has generic mount paths, never a usable protected path.

No-flip and full classifier runs require substantial GPU time; `--help` and config
validation are not training. Calibration, operating-point, and acquisition scripts
post-process existing predictions without retraining image classifiers. Intervals
condition on the fitted ensemble; seed variation is a separate descriptive quantity.

The archived five-image cap groups the sorted manifest by encounter and takes
`head(5)` in every partition, affecting both training and evaluation. It is neither
random frame selection nor an inference-only ablation. The published single-arm
cap does not test an augmentation-by-image-count interaction.

GAN and structural-fidelity jobs were pending when this release was prepared.
This release does not claim their completion or contain their results.
