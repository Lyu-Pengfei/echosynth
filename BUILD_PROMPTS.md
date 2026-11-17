# Build Prompts for `echosynth`

> Historical provenance note: this file preserves prompts used while constructing the original
> encounter-level workflow. The released reference implementation now uses unique-patient
> partitioning and image-to-encounter-to-patient aggregation, as documented in `README.md`,
> `MODEL_CARD.md`, and `PROJECT_SPEC.md`.

Hand these to a local coding agent **in order**. Each prompt is self-contained and ends with
acceptance criteria the agent must satisfy before moving on.

**How to use**

1. Put `PROJECT_SPEC.md` in the repo root and keep it open in the agent's context.
2. Paste **P0** once as the system prompt / `CLAUDE.md` / `.cursorrules` / `AGENTS.md`.
3. Run P1 → P13 one at a time. After each, run `make test` and review the diff before
   continuing. Do not batch phases — the failure modes compound.
4. If the agent deviates from the spec, say *"re-read §X of PROJECT_SPEC.md and fix"* rather
   than describing the fix yourself.

---

## P0 — Ground rules (system prompt, paste once)

```
You are building `echosynth`, the reference implementation of the paper "Synthetic
Echocardiograms from Diffusion Models in Rare Cardiovascular Disease".

PROJECT_SPEC.md in the repo root is the authoritative contract. Read it before every task.
Where this prompt and the spec disagree, the spec wins. Where the spec is silent, ASK — do
not invent behaviour and do not silently pick a default.

Environment
- Python 3.11. PyTorch >= 2.1. Type hints on every public function. Ruff + mypy clean.
- Dependencies pinned in requirements.txt. The Keras/TensorFlow view classifier lives in a
  SEPARATE environment (requirements-view.txt). Never import tensorflow and torch in one
  process.
- No network access at runtime. Model weights and data are supplied as local paths.

Non-negotiable engineering rules
1. NO DATA LEAKAGE. Splitting is at the encounter level (patient level for the
   real-vs-synthetic discriminator). Generators train on the train split only. Synthetic
   images never enter validation or test. Every entry point calls
   assert_manifest_invariants() first.
2. THE TEST SET IS TOUCHED ONCE. Model selection, epoch selection, synthetic-sample-count
   selection, and every hyperparameter choice use the validation split. Enforce this in
   code, not by convention.
3. NO INVENTED NUMBERS. Never hard-code, mock, or "approximate" a result to make a test
   pass. If something cannot run, raise NotImplementedError with an explanatory message.
4. NO SILENT FAILURES. `except: pass` is banned. Every skipped or rejected file is logged
   with a reason code and counted in a summary.
5. DETERMINISM. One seed function seeds python/numpy/torch/cuda. Every run writes its fully
   resolved config, git SHA, and seed alongside its results.
6. NO PHI IN THE REPO. No DICOMs, PNGs, checkpoints, or manifests with real identifiers get
   committed. A pre-commit hook enforces this.
7. SCIENCE LIVES IN src/echosynth/. scripts/*.py are argparse shells under 60 lines.

Style
- Small, testable, pure-where-possible functions. Dataclasses for structured returns.
- Docstrings cite the paper section or line numbers they implement, e.g.
  """Implements the 8-feature QC screen (paper §2.3, lines 147-165)."""
- Write the test in the same commit as the code.

Provenance and licensing
- github.com/raventan95/echo-view-classifier is MIT: adapted code is allowed, and its notice
  must appear in THIRD_PARTY_NOTICES.md.
- github.com/rhenaog/hcm_duke_model has NO LICENSE: do not copy its source. Reimplement the
  behaviour (encounter-level aggregation, best-checkpoint-on-val-AUC, background masking)
  independently.
- github.com/MuhangTian/synthetic-medical-image vendors lucidrains' denoising-diffusion-pytorch
  (MIT upstream). Reuse the structure; note the upstream attribution.

Workflow
- After each phase: `make lint && make test`. Report what you built, what you assumed, and
  what you could not verify. Never claim a component works if you have not run it.
```

---

## P1 — Scaffold and tooling

```
Create the repository skeleton for `echosynth` exactly as specified in PROJECT_SPEC.md §2.

Deliverables:
- pyproject.toml (PEP 621, package name `echosynth`, src layout, ruff + mypy config)
- requirements.txt (pinned) and requirements-view.txt (TF/Keras view classifier env)
- environment.yml
- Makefile with targets: install, lint, test, test-fast, ingest, qc, preprocess, views,
  splits, gen, sample, train, experiment, fidelity, report, clean
- .gitignore covering data/, results/, wandb/, *.dcm, *.png, *.pt, *.h5, *.pkl
- .pre-commit-config.yaml with ruff, ruff-format, and a custom `no-phi` hook that rejects
  staged files under data/, any *.dcm/*.png/*.pt/*.h5, and files containing 8+ consecutive
  digits (MRN-like) or ISO dates in a filename
- Every package/module file from the §2 tree, each with a real module docstring and NotImplementedError
  stubs for the public functions named in §4 — correct signatures and type hints, no bodies yet
- README.md: install, the 12-step pipeline, and a "where the numbers come from" table
  pointing at PROJECT_SPEC.md §7
- THIRD_PARTY_NOTICES.md with the raventan95 MIT notice and the lucidrains attribution
- docs/DECISIONS.md, pre-filled with the six open questions from PROJECT_SPEC.md §10, each
  marked UNRESOLVED

Also implement for real (not stubs):
- src/echosynth/utils/seed.py:   set_seed(seed: int) -> None
- src/echosynth/utils/logging.py: get_logger(name) with a RejectionCounter class that tallies
  reason codes and prints a summary table
- src/echosynth/utils/config.py: load_config(path, overrides: list[str]) -> Config, supporting
  a `_base:` key for inheritance and `--set a.b=c` overrides; dataclass validation; and
  save_resolved(config, out_dir) which also records the git SHA
- src/echosynth/utils/io.py, checkpoint.py: atomic writes, save/load checkpoint with metadata

Acceptance:
- `pip install -e .` succeeds; `make lint` is clean; `python -c "import echosynth"` works
- `pytest tests/` collects and passes (stubs are xfail/skip, not silently passing)
- `make test` runs in under 30 seconds at this stage
Do NOT implement any pipeline stage yet.
```

---

## P2 — Manifests, splits, and the leakage guard

```
Implement PROJECT_SPEC.md §3 and §4.7's TestSetGuard. This is the backbone — everything else
depends on it, so it comes before any imaging code.

Files: src/echosynth/data/manifest.py, src/echosynth/data/splits.py,
       src/echosynth/eval/guard.py

Implement:
- A ManifestSchema dataclass with the exact §3.1 columns, dtypes, and enums; read_manifest()
  and write_manifest() that validate on both sides.
- assert_manifest_invariants(df) enforcing all five invariants in §3.4, raising
  LeakageError with a message naming the offending ids.
- make_encounter_splits(df, ratios=(0.70, 0.15, 0.15), seed=2024, stratify_by="group")
  -> pd.DataFrame. Splits at the ENCOUNTER level, stratified by group, deterministic
  under seed. All images of an encounter land in one partition.
- TestSetGuard: a context manager that permits exactly one test-set evaluation per run,
  logs the access with a timestamp and caller, and raises TestSetReuseError on a second.

Tests (tests/unit/test_splits.py, test_manifest.py):
- A synthetic manifest of 4,697 encounters with the §3.4 group proportions splits at seed
  2024 into partitions matching the Table 1 counts (216/46/47, 1542/330/331, 1529/328/328).
- Splitting twice at the same seed is identical; a different seed differs.
- An encounter deliberately spread across two splits raises LeakageError.
- A synthetic row with split="test" raises LeakageError.
- TestSetGuard raises on second entry.

Acceptance: all tests pass; no imaging dependency is imported by these modules.
```

---

## P3 — DICOM ingest

```
Implement PROJECT_SPEC.md §4.1 in src/echosynth/data/dicom_ingest.py, driven by
scripts/00_ingest_dicom.py.

Requirements:
- pydicom with pylibjpeg + gdcm fallbacks for compressed transfer syntaxes.
- is_standard_acquisition(): SOP Class UID allowlist in configs/data/duke.yaml, not in code.
- has_color_doppler(): check UltrasoundColorDataPresent plus photometric-interpretation and
  palette hints; document each heuristic in the docstring.
- extract_first_frame(): frame index 0 always. Docstring must state this is a predefined
  rule, NOT a diagnostically-optimal frame selection (paper line 145).
- ingest(): parallel over files with a configurable worker count, writes PNGs and the master
  manifest, and returns a summary of rejections by reason code. NEVER swallow an exception —
  catch, log with a reason code, count it, continue.
- Reason codes: UNREADABLE, NONSTANDARD_SOP, COLOR_DOPPLER, NO_PIXEL_DATA, DECODE_FAILED,
  DUPLICATE_ID.

CLI: scripts/00_ingest_dicom.py --root PATH --out-dir PATH --manifest PATH
     --config configs/data/duke.yaml [--workers N] [--dry-run]

Tests: build 6 synthetic DICOM files in tests/fixtures/ with pydicom (one valid multiframe,
one colour-Doppler, one non-standard SOP, one with no pixel data, one corrupt, one duplicate)
and assert each lands in the right reason bucket and that exactly one row reaches the manifest.

Acceptance: tests pass; --dry-run touches nothing on disk; the rejection summary prints.
```

---

## P4 — Quality control

```
Implement PROJECT_SPEC.md §4.2 in src/echosynth/data/qc.py, driven by scripts/01_run_qc.py.

Compute the eight features in exactly the spec's order and names. Base-2 Shannon entropy over
the 256-bin histogram; scipy Fisher kurtosis; Canny thresholds from config. Z-standardize
across the dataset, set qc_flag = any(|z| > 3), write manifests/qc_review_queue.csv with the
raw features, the z-scores, and an empty `reviewer_decision` column.

CRITICAL (paper lines 159-165): the flag is a HIGH-SENSITIVITY SCREEN, never an automatic
exclusion. qc_excluded is set only by `--apply-review reviewed.csv`, which requires a
non-empty reviewer_decision per flagged row. Flagged-but-unconfirmed rows are RETAINED.
If the agent is tempted to auto-exclude, it has misread the paper.

CLI: scripts/01_run_qc.py --manifest PATH --config PATH [--apply-review PATH]

Tests: hand-built images with known answers — an all-black frame (nonblack_ratio == 0), a
uniform-noise frame (entropy near 1.0 normalized), a solid frame (std == 0, entropy == 0),
a synthetic wedge (sector_coverage within tolerance of the drawn fraction). Plus: an
outlier frame is flagged but NOT excluded until a review CSV is applied.

Acceptance: tests pass; running QC twice is idempotent.
```

---

## P5 — Registration, masking, preprocessing

```
Implement PROJECT_SPEC.md §4.3 in registration.py, masking.py, preprocess.py, driven by
scripts/02_preprocess.py.

- detect_sector(): find the ultrasound wedge (apex, inner/outer radius, opening angle) from
  the non-zero support. Return a Sector dataclass. Handle the failure case explicitly — return
  None and log SECTOR_NOT_FOUND rather than guessing.
- fit_circular_model(): least-squares circle fit to the outer arc -> CircleFit.
- The reference geometry is the MEDIAN CircleFit over the TRAIN SPLIT ONLY, computed once and
  cached to results/reference_sector.json. Val, test, and synthetic images all align to that
  cached reference. Recomputing it from a non-train split is a leakage bug.
- align_to_reference(): translation + rotation + affine to the reference frame.
- sector_mask(): zero everything outside the wedge, removing text overlays, ECG traces, and
  device annotations.
- preprocess_image(): detect -> fit -> align -> mask -> resize(112) -> center-crop(112),
  output uint8 single-channel. This exact order.

CLI: scripts/02_preprocess.py --manifest PATH --out-dir PATH --config PATH [--build-reference]

Tests: draw a synthetic wedge at a known offset and rotation; assert registration recovers it
within tolerance; assert masking is idempotent; assert the output is (112,112) uint8 with zero
pixels outside the mask; assert that calling --build-reference on a manifest containing val
rows raises.

Acceptance: tests pass; a visual QA contact sheet is written to results/preprocess_qa.png.
```

---

## P6 — View classifier

```
Implement PROJECT_SPEC.md §4.4 in src/echosynth/views/, driven by scripts/03_classify_views.py.

Adapted from github.com/raventan95/echo-view-classifier (MIT — keep the notice). Label map:
{plax:0, psax-av:1, psax-mv:2, psax-ap:3, a4c:4, a5c:5, a3c:6, a2c:7}. Input contract fixed by
the pretrained VGG16 weights: 224x224 RGB.

Two modes:
- mode=onnx (PREFERRED): export_onnx.py converts the .h5 once; inference runs under
  onnxruntime in the main env.
- mode=keras: runs as a SUBPROCESS in the view env; port the deprecated predict_generator
  to model.predict.

Either way, upscale the 112x112 grayscale preprocessed image to 224x224 and replicate to 3
channels FOR VIEW CLASSIFICATION ONLY, then write view_pred and view_conf back into the
master manifest. Never feed 224x224 RGB to the diffusion model or CompactCNN.

CLI: scripts/03_classify_views.py --manifest PATH --weights PATH --mode {onnx,keras}

Tests: mock the model to return a fixed softmax and assert the label mapping, the confidence
column, and manifest round-tripping. Skip (do not fail) if weights are absent.

Acceptance: tests pass; a missing-weights run exits with a clear message and a download URL,
not a stack trace. Confirm in the README that the PRIMARY endpoint is all-view and does not
depend on this stage.
```

---

## P7 — Class-conditional diffusion model

```
Implement PROJECT_SPEC.md §4.5 in src/echosynth/generative/.

Start from the structure of github.com/MuhangTian/synthetic-medical-image model/ddpm.py
(vendored lucidrains denoising-diffusion-pytorch), but that version is UNCONDITIONAL. You
must build the classifier-free-guidance variant.

unet_cfg.py — ClassConditionalUnet with:
- a learned class embedding (nn.Embedding(num_classes, dim)) added to the time embedding
- a learned null-class embedding, dropped in with probability cond_drop_prob during training
- forward(x, time, classes, cond_drop_prob=None)
- forward_with_cond_scale(x, time, classes, cond_scale) implementing
  eps_tilde = eps(x,t,null) + s * (eps(x,t,y) - eps(x,t,null)),  s >= 1   [paper line 192]
- attention at higher resolutions only, full_attn=(False, False, False, True)

diffusion.py — GaussianDiffusion with conditional p_sample/ddim_sample accepting `classes`
and `cond_scale`. Fixed values from the paper: timesteps=1000, sampling_timesteps=500,
beta_schedule='sigmoid', objective='pred_noise', image_size=112, channels=1.
Because sampling_timesteps < timesteps this takes the DDIM path — log which sampler ran.

dataset.py — LabeledEchoDataset(manifest, split, label_col, class_map, image_size=112,
augment_horizontal_flip=True) returning (tensor, class_idx). It must call
assert_manifest_invariants and raise if any non-train row is present.

trainer.py — accelerate + EMA + AMP, batch_size=16, lr=8e-5, checkpoint and class-balanced
preview grid every save_and_sample_every steps, optional wandb. FID off by default (the
paper reports pooled-histogram divergences, not FID).

sample.py — generate(ckpt, n_per_class, cond_scale, sampling_steps, out_dir, seed,
manifest_out) writing PNGs plus manifests/synthetic_<task>.csv per §3.2. Every synthetic row
gets split="train" and lives in a directory tree physically separate from real images.

CLI: scripts/05_train_diffusion.py, scripts/06_sample_synthetic.py, both config-driven.

Tests:
- shape test: forward pass at 112x112, batch 4, num_classes 3
- cond_scale=1.0 reproduces the plain conditional prediction; cond_scale>1 moves the
  prediction away from the unconditional one in the expected direction
- LEAKAGE TEST: plant a split="test" row in the dataset manifest, assert the trainer raises
- a 20-step training run on tests/fixtures/tiny/ completes and writes a loadable checkpoint
- sampling with a fixed seed is reproducible byte-for-byte

Acceptance: all tests pass on CPU in under 3 minutes. Do NOT start a real training run.
```

---

## P8 — CompactCNN classifier and the two-step framework

```
Implement PROJECT_SPEC.md §4.6 in src/echosynth/classify/.

models.py — CompactCNN: four [Conv3x3 -> ReLU -> MaxPool2] blocks (channels 1->32->64->128->256,
config-driven) -> AdaptiveAvgPool2d(1) -> Dropout(p) -> Linear(256, 2). Adam +
CosineAnnealingLR.

transforms.py — TRAIN ONLY: random horizontal flip, rotation up to +/-10 degrees, random
brightness and contrast jitter (paper lines 208-210). Val and test get resize/normalize only.
Make it structurally impossible to apply train transforms at eval time.

engine.py — train/evaluate loops. Model selection on VALIDATION ENCOUNTER-LEVEL AUROC (not
image-level, not loss). Early stopping with patience. Checkpoint best + last. The evaluate()
function must accept a TestSetGuard when the split is "test".

two_step.py — TwoStepClassifier per §4.6: Stage 1 (CA union LVH) vs GEN, Stage 2 CA vs LVH.
Return the TwoStepResult dataclass. Composition is config-driven:
  product      -> p_ca = p_abnormal * p_ca_given_abnormal   (DEFAULT)
  stage2_only  -> p_ca = p_ca_given_abnormal
  logit_sum    -> p_ca = sigmoid(logit(p_abnormal) + logit(p_ca_given_abnormal))
Compose PER IMAGE, then average within encounter (spec §10 decision 2). Write the chosen
default into docs/DECISIONS.md and mark decisions 1-3 RESOLVED.

Also reimplement, independently of rhenaog/hcm_duke_model (unlicensed — do not copy its code):
encounter-level probability averaging, and best-checkpoint-on-validation-AUC selection.

Tests: forward/shape tests; a train transform never fires in eval mode; the two-step
composition matches a hand-computed example for all three modes; the model overfits a
20-sample fixture to training AUROC > 0.95 within 30 epochs (a learning-capacity smoke test).

Acceptance: tests pass on CPU under 2 minutes.
```

---

## P9 — Metrics, DeLong, and aggregation

```
Implement PROJECT_SPEC.md §4.8 in src/echosynth/eval/.

aggregate.py — to_encounter_level(df, prob_col, by="encounter_id"): mean of image-level
probabilities within encounter. Docstring states this is the only reported unit (paper
lines 240-244) and explains why (the independence assumption is violated otherwise).

metrics.py — auroc, auprc. Thin, tested wrappers over sklearn.

delong.py — the fast DeLong algorithm (Sun & Xu 2014), implemented from the algorithm, with
delong_auc_variance, delong_ci, and delong_paired_delta_ci returning a DeltaResult dataclass
(delta, ci_low, ci_high, z, p_value).

AUPRC gets a point estimate only — no confidence interval (paper line 239). If asked for one,
raise NotImplementedError with that citation.

Tests (this module is load-bearing, so be thorough):
- AUROC of a perfect ranker == 1.0; of a random constant == 0.5
- DeLong AUC agrees with sklearn's roc_auc_score to 1e-10 on 10 random vectors
- DeLong variance matches a hand-computed value on a small fixed vector
- The paired test on identical predictions gives delta == 0 and p == 1
- Round-trip check against the paper: given the reported Duke CA numbers
  (0.883 vs 0.924, delta 0.041 with 95% CI 0.013-0.069), the implied paired SE is
  approximately 0.0143 — assert your delta CI construction reproduces that CI from that SE
- to_encounter_level: 3 images in one encounter with probs 0.2/0.4/0.9 -> 0.5, and label
  consistency within encounter is asserted

Acceptance: all tests pass. This is the module most likely to be quietly wrong — do not move
on until every test is green.
```

---

## P10 — Training paradigms and the experiment runner

```
Implement PROJECT_SPEC.md §4.7 in src/echosynth/classify/paradigms.py, driven by
scripts/08_run_experiment.py.

Four paradigms: TRTR, TSTR, TRSTR, BOOTSTRAP, per the §4.7 table.

Synthetic-count selection (paper lines 214-219):
- grid {100, 200, ..., N_max} from config (default stop 3000)
- for each count: train on train, score on VALIDATION encounter-level AUROC
- take argmax, then evaluate that single configuration ONCE on test through TestSetGuard
- the test split selects NOTHING: not the count, not the model, not the epoch, not any
  hyperparameter. If you find yourself reading the test split during selection, stop.

Bootstrap baseline (lines 245-250): resample real MINORITY-class TRAINING rows with
replacement; the number added equals the number of synthetic samples in the matched TRSTR
run; majority class untouched; a fresh bootstrap draw per training run; every other setting
identical to the TRSTR arm.

run_experiment(config) -> ExperimentResult writes results/runs/<run_id>/ containing
config.yaml (resolved), git SHA, seed, metrics.json (per-paradigm AUROC/AUPRC with DeLong
CIs plus the paired delta vs TRTR), the selection curve over the synthetic grid, and log.txt.

CLI: scripts/08_run_experiment.py --config configs/experiment/duke_two_step_allview.yaml

Tests:
- an end-to-end run on tests/fixtures/tiny/ over a 2-point grid completes and writes all
  artifacts
- the bootstrap arm adds exactly as many rows as the matched synthetic arm
- selection provably never reads the test split (assert via a spy on the dataloader)
- TestSetGuard raises if run_experiment tries a second test evaluation

Acceptance: tests pass in under 5 minutes on CPU.
```

---

## P11 — Fidelity, discriminator, and survey analysis

```
Implement PROJECT_SPEC.md §4.9 and §4.10.

fidelity.py — reproduce the paper's pooled-grayscale-histogram procedure EXACTLY, quirks
included, because the reported numbers depend on them (lines 269-283):
  1. load with Pillow, convert to "L"
  2. accumulate a 256-bin numpy histogram over [0, 256)
  3. DO NOT resize and DO NOT weight images equally — each image contributes in proportion
     to its pixel count. This looks like a bug. It is what the paper did. Keep it, and say
     so in the docstring.
  4. add 1e-12 to every bin, then normalize
  5. report KL(real||syn), KL(syn||real), Jensen-Shannon DIVERGENCE, and JS DISTANCE
     (= sqrt of the divergence) as four separate numbers
  6. Brier_dist = mean_b (p_b - q_b)^2 ; RMSE_dist = sqrt(Brier_dist)

embedding.py — t-SNE and UMAP over the same feature matrix, real vs synthetic coloured by
class. These are QUALITATIVE diagnostics: the function must not return or plot any p-value
or test statistic.

discriminator.py — real-vs-synthetic binary CNN. PATIENT-level 70/15/15 split (stronger than
encounter level). Selection on val, one evaluation on test, AUROC with DeLong CI. The
docstring notes that 0.5 means no separability and that the observed value is read alongside
its CI, the partition design, and class composition (lines 286-293).

survey.py — item-wise denominators (every percentage carries its own n); normalized entropy
H_i = -sum_k p_ik log2 p_ik, H_norm = H_i / log2(K) with K the categories FOR THAT ITEM,
averaged within the real and synthetic groups; paired responses recoded post-unblinding into
{prefers_real, prefers_synthetic, cannot_decide}. Responses are NOT independent (each
evaluator rates many images) — emit descriptive statistics only, and raise if asked for a
confirmatory test.

Tests:
- KL of a distribution with itself == 0; JS distance == sqrt(JS divergence) on random pairs
- a hand-computed 4-bin example matches by hand
- normalized entropy of a uniform response distribution == 1.0; of a degenerate one == 0.0
- discriminator raises if the same patient appears in two splits
- survey percentages use per-item denominators (plant a missing response and check n drops
  for that item only)

Acceptance: tests pass; scripts/09_fidelity.py, 10_discriminator.py, 11_survey.py all run on
the tiny fixture.
```

---

## P12 — EchoNet adapters

```
Implement PROJECT_SPEC.md §4.11 in src/echosynth/data/echonet.py.

EchoNet-Dynamic: decode videos -> first frame -> the same S3 preprocessing -> LVEF buckets
normal (EF >= 55), mild (40 <= EF < 55), severe (EF < 40). Log the resulting counts and warn
if they deviate from the expected 6961 / 805 / 1246 by more than 1%.

EchoNet-LVH: drop videos missing IVSd or LVPWd (expect 349 dropped, 11651 retained);
max_wt = max(IVSd, LVPWd); tau = 90th percentile of max_wt computed FROM THE TRAINING SPLIT
ONLY; assert tau == 1.3274 within 1e-3 on the official partition; apply tau unchanged to val
and test. Assert the class counts: train 9152/1019, val 1037/104, test 210/129.

Emit a LOUD runtime warning that the official test partition is roughly 38% minority against
roughly 10% in train. That is a property of the released split, not a bug in your code — but
nobody should read AUPRC across those partitions as comparable without noticing it.

Both adapters produce manifests matching the §3.3 schema and reuse the same splits/QC/
preprocessing code paths — no forked logic.

CLI: extend scripts/00_ingest_dicom.py or add scripts/00b_ingest_echonet.py.

Tests: a fixture CSV of 50 synthetic rows reproduces the threshold logic exactly; a row
missing LVPWd is dropped; tau computed from a manifest containing val rows raises.

Acceptance: tests pass.
```

---

## P13 — Reporting, SLURM, and the end-to-end smoke test

```
Final phase. Implement PROJECT_SPEC.md §12, §8.2, and §9.

report/tables.py — regenerate Table 1 (encounter split counts), Table 2 (expert evaluation),
and Table 3 (all-view two-step performance with AUROC, bootstrap AUROC, real-plus-synthetic
AUROC, and delta-AUROC with 95% CIs) from results/runs/*/metrics.json. Output both tidy CSV
and formatted markdown. Numbers come from artifacts on disk — never from a literal in the
code.

report/figures.py — Figure 3 (t-SNE + UMAP side by side) and Figure 4 (ROC and PR curves,
TRTR vs TSTR, with AUROC and AUPRC in the legend). Matplotlib only, no seaborn styling
dependency, 300 dpi, colourblind-safe palette.

shell/slurm/ — parameterized templates per §9 (train_diffusion.sh, sample.sh,
train_classifier.sh as an array job over the synthetic grid, preprocess.sh as a CPU array).
Read partition, account, email, and paths from shell/slurm/cluster.env, which is gitignored
with a committed cluster.env.example. No hard-coded cluster identifiers anywhere.

tests/integration/test_end_to_end.py — the §8.2 smoke test. Procedurally generate ~60
112x112 wedge images across 3 classes and 20 encounters into tests/fixtures/tiny/, then run:
splits -> 200-step diffusion train -> sample 10 per class -> CompactCNN 3 epochs ->
encounter-level AUROC -> report. Must finish in under 5 minutes on CPU. Assert every expected
artifact exists and that metrics.json is well-formed.

Also add tests/integration/test_reproduction.py: a table of the PROJECT_SPEC.md §7 target
values, skipped by default with a clear message, runnable via `pytest -m reproduction` when
real data is mounted. It must ASSERT AGAINST the targets, never hard-code them as outputs.

Finally, update README.md with the full 12-step runbook, and update docs/DECISIONS.md so
every one of the six §10 questions is either RESOLVED with its rationale or still marked
UNRESOLVED with the blocking question stated.

Acceptance: `make test` green; `make report` produces tables and figures from the smoke run;
`bash shell/slurm/train_diffusion.sh --dry-run` prints a valid sbatch invocation.
```

---

## Suggested prompting cadence

| Phase | Est. agent time | Human review focus |
|---|---|---|
| P1 | 20 min | Does the tree match §2? Is `no-phi` actually blocking? |
| P2 | 30 min | **Table 1 counts reproduce exactly.** Leakage tests genuinely fail on bad input. |
| P3–P5 | 2–3 h | Sector detection on your real frames. Look at `preprocess_qa.png`. |
| P6 | 30 min | ONNX export matches Keras output on the sample images. |
| P7 | 2 h | Preview grids after ~1k steps — does anything look like a heart? |
| P8 | 1 h | Two-step composition matches your intended reading (§10 Q1). |
| P9 | 1 h | **DeLong against a known-good implementation.** Highest-risk module. |
| P10 | 1.5 h | Trace the test-set access log; confirm exactly one access per run. |
| P11 | 1.5 h | Histogram quirks preserved. Does it reproduce KL 0.019 / 0.017? |
| P12 | 45 min | τ = 1.3274 assertion fires on the real split. |
| P13 | 1 h | Tables regenerate from disk, not from literals. |

**Before P1, resolve §10 question 1 (two-step composition).** It changes the primary endpoint
and is the one ambiguity the agent cannot settle from the manuscript alone.
