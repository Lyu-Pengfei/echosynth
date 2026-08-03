# `echosynth` — Project Specification

Reference implementation for *Synthetic Echocardiograms from Diffusion Models in Rare
Cardiovascular Disease* (Duke CA cohort; EchoNet-Dynamic; EchoNet-LVH).

This document is the **contract**. `BUILD_PROMPTS.md` turns it into an ordered set of
prompts for a local coding agent. Nothing in the build prompts may contradict this file.

---

## 0. Scope and non-goals

**In scope**

| # | Capability | Paper reference |
|---|---|---|
| S1 | DICOM ingest → first-frame PNG + manifest | §2.3 lines 140–146 |
| S2 | 8-feature automated QC screen (\|Z\|>3 flag, manual confirm) | §2.3 lines 147–165 |
| S3 | Resize 112×112, center crop, sector registration, sector masking | §2.3 lines 166–173 |
| S4 | View classification (PLAX / A2C / A3C / A4C tagging) | Fig. 1a |
| S5 | Class-conditional DDPM with classifier-free guidance | §2.4 |
| S6 | `CompactCNN` classifier + two-step CA framework | §2.5, §2.5.2 |
| S7 | Training paradigms: TRTR / TSTR / TRSTR / bootstrap | §2.5.1, §2.6 |
| S8 | Encounter-level AUROC/AUPRC, DeLong CIs, paired ΔAUROC CI | §2.6 |
| S9 | Fidelity: t-SNE/UMAP, pooled-histogram divergences, real-vs-synthetic discriminator | §2.7 |
| S10 | Expert-survey analysis (normalized entropy, Table 2) | §2.8, §3.1.2 |
| S11 | EchoNet-Dynamic (LVEF) and EchoNet-LVH (wall thickness) adapters | §2.2.2, §2.2.3 |
| S12 | Table/figure regeneration (Tables 1–3, Figures 3–4) | §3 |

**Explicit non-goals** (match the paper's stated limitations, lines 453–460)

- No video/temporal diffusion. First frame only.
- No resolution above 112×112.
- No multimodal (clinical variable) fusion.
- No memorization / nearest-neighbour / formal privacy audit **in v1** — but leave the
  `eval/privacy.py` module stub in place, since it is named as future work (line 463).
- No PHI in the repo. Ever. Enforced by a pre-commit hook (§8.4).

---

## 1. What each reference repo contributes

| Repo | License | What we take | What we must change |
|---|---|---|---|
| `MuhangTian/synthetic-medical-image` | none stated | Overall shape: `model/`, `preprocess/`, `train.py`, `sweep/*.yaml`, `shell/*.sh` SLURM pattern. `model/ddpm.py` is a vendored copy of lucidrains `denoising-diffusion-pytorch` (+ `attend.py`, `fid_evaluation.py`). | Its `Unet`/`GaussianDiffusion` are **unconditional**. We need the classifier-free-guidance variant (`denoising_diffusion_pytorch/classifier_free_guidance.py` upstream) so the U-Net accepts `classes` and `cond_drop_prob`. Its `Dataset` returns an image only — we need `(image, class_idx)`. Its `train_test_split` moves files on disk by integer index — replace entirely with an **encounter-level** manifest split. |
| `raventan95/echo-view-classifier` | MIT (attribution required) | `classify.py` inference pattern; the 8-view label map `{plax, psax-av, psax-mv, psax-ap, a4c, a5c, a3c, a2c}`; the 224×224 RGB input contract; pretrained VGG16 `.h5` weights (Dropbox, ~0.939 accuracy). | TF1.12/Keras `predict_generator` is deprecated. Isolate in its own env or export to ONNX (§4.4). Never let TF and PyTorch share a process. |
| `rhenaog/hcm_duke_model` | **no license** | *Ideas only, no copied code.* Namely: encounter/study-level probability aggregation before AUC; select best checkpoint on validation AUC; masked-background loading; `save_checkpoint`/`load_checkpoint` shape; `r2plus1d_18` as the optional future video backbone. | The published script is a flat top-level script with hard-coded paths, an `r2plus1d_18` video model, and a validation loop with a duplicated `valid_running_loss` accumulation bug. Reimplement from scratch as library code. |

> **Legal note for the build agent:** `hcm_duke_model` carries no license. Do not copy its
> source. Reimplement the behaviour independently. Reproduce the MIT notice for
> `echo-view-classifier` in `THIRD_PARTY_NOTICES.md`.

---

## 2. Repository layout

```
echosynth/
├── README.md
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml                  # package = echosynth, PEP 621, ruff + mypy config
├── requirements.txt                # pinned; requirements-view.txt for the TF env
├── environment.yml
├── Makefile                        # make ingest | qc | preprocess | views | gen | train | eval | report
├── .gitignore                      # data/, results/, *.dcm, *.png, *.pt, wandb/
├── .pre-commit-config.yaml
│
├── configs/
│   ├── base.yaml
│   ├── data/{duke,echonet_dynamic,echonet_lvh}.yaml
│   ├── diffusion/{duke_ca,duke_lvh,duke_general,echonet_dynamic_lvef,echonet_lvh_wt}.yaml
│   ├── classifier/{compactcnn,r2plus1d}.yaml
│   └── experiment/
│       ├── duke_two_step_allview.yaml        # PRIMARY ENDPOINT
│       ├── duke_tstr_diagnostic.yaml         # secondary
│       ├── duke_bootstrap_baseline.yaml      # secondary
│       ├── echonet_lvh_wallthickness.yaml    # secondary
│       └── fidelity_echonet_dynamic.yaml     # exploratory
│
├── src/echosynth/
│   ├── __init__.py
│   ├── utils/          seed.py logging.py io.py checkpoint.py config.py
│   ├── data/           dicom_ingest.py qc.py registration.py masking.py preprocess.py
│   │                   manifest.py splits.py datasets.py echonet.py
│   ├── views/          view_classifier.py export_onnx.py
│   ├── generative/     unet_cfg.py diffusion.py dataset.py trainer.py sample.py
│   ├── classify/       models.py transforms.py engine.py two_step.py paradigms.py
│   ├── eval/           metrics.py delong.py aggregate.py fidelity.py discriminator.py
│   │                   embedding.py survey.py privacy.py
│   └── report/         tables.py figures.py
│
├── scripts/            00_ingest_dicom.py 01_run_qc.py 02_preprocess.py
│                       03_classify_views.py 04_make_splits.py
│                       05_train_diffusion.py 06_sample_synthetic.py
│                       07_train_classifier.py 08_run_experiment.py
│                       09_fidelity.py 10_discriminator.py 11_survey.py
│                       12_make_report.py
│
├── shell/              slurm/*.sh (gpu + cpu templates), local/*.sh
├── tests/              unit/ integration/ fixtures/
├── data/               .gitkeep only — everything else gitignored
└── results/            runs/<run_id>/{ckpt,samples,metrics.json,config.yaml,log.txt}
```

**Rule:** every `scripts/*.py` is a thin argparse shell (< 60 lines) over
`src/echosynth/`. No science lives in `scripts/`.

---

## 3. Data contracts

Stages communicate only through CSV manifests. This is the single most important
design constraint — get it wrong and every downstream stage silently breaks.

### 3.1 `manifests/duke_master.csv`

| column | type | notes |
|---|---|---|
| `image_id` | str | stable hash, primary key |
| `patient_id` | str | de-identified surrogate |
| `encounter_id` | str | **unit of splitting and of final evaluation** |
| `group` | enum | `CA` \| `LVH` \| `GEN` |
| `study_date` | date | CA: 2004-06→2024-03; controls: 2015-01→2023-12 |
| `dicom_path` | path | source; never committed |
| `png_path` | path | post-preprocess 112×112 grayscale PNG |
| `sop_class_uid` | str | used for non-standard-acquisition exclusion |
| `color_doppler` | bool | from ultrasound colour-flow tags |
| `view_pred` | enum | from S4; nullable |
| `view_conf` | float | softmax max |
| `qc_flag` | bool | any \|Z\| > 3 |
| `qc_features` | json | the 8 raw features |
| `qc_excluded` | bool | **only** true after manual confirmation |
| `qc_reason` | str | free text |
| `split` | enum | `train` \| `val` \| `test` |

### 3.2 `manifests/synthetic_<task>.csv`

`image_id, task, class_label, guidance_scale, sampling_steps, generator_run_id, png_path, seed`

Synthetic rows carry `split = "train"` **always**, and are stored in a physically
separate directory tree from real images.

### 3.3 EchoNet manifests

- `echonet_dynamic.csv`: `video_id, frame_path, ef, lvef_class ∈ {normal, mild, severe}, split`
  Class rule: `normal` EF ≥ 55 (n≈6,961), `mild` 40 ≤ EF < 55 (n≈805), `severe` EF < 40 (n≈1,246).
- `echonet_lvh.csv`: `video_id, frame_path, ivsd, lvpwd, max_wt = max(ivsd, lvpwd), wt_class, split`
  `wt_class = 1 if max_wt > τ`, with **τ computed from the training split only** as the 90th
  percentile, and asserted equal to `1.3274` (tolerance 1e-3) on the official partition.

### 3.4 Invariants (assert, don't hope)

```python
# src/echosynth/data/manifest.py
def assert_manifest_invariants(df: pd.DataFrame) -> None: ...
```

1. `encounter_id → split` is a function (no encounter spans two splits).
2. `patient_id → split` is a function (stronger; required for the discriminator in §9.3).
3. No `png_path` appears in more than one split.
4. Synthetic images never appear in `val` or `test`.
5. Duke encounter counts reproduce Table 1 exactly at seed `2024`:

   | group | train | val | test | total |
   |---|---|---|---|---|
   | CA | 216 | 46 | 47 | 309 |
   | GEN | 1,542 | 330 | 331 | 2,203 |
   | LVH | 1,529 | 328 | 328 | 2,185 |
   | **total** | **3,287** | **704** | **706** | **4,697** |

---

## 4. Stage specifications

### 4.1 S1 — DICOM ingest (`data/dicom_ingest.py`)

```python
def scan_archive(root: Path) -> pd.DataFrame: ...
def is_standard_acquisition(ds: pydicom.Dataset) -> bool: ...      # SOP Class UID allowlist
def has_color_doppler(ds: pydicom.Dataset) -> bool: ...            # UltrasoundColorDataPresent + photometric hints
def extract_first_frame(ds: pydicom.Dataset) -> np.ndarray: ...    # deterministic rule, NOT "best" frame
def ingest(root, out_dir, manifest_out, n_workers) -> pd.DataFrame: ...
```

- Use `pydicom` with `pylibjpeg` / `gdcm` for compressed transfer syntaxes.
- Multi-frame → frame index 0, always. Document that this is a *predefined rule, not a
  diagnostic-phase selection* (paper line 145).
- Log every rejection with a reason code. A silent `try/except: pass` (as in the reference
  repo's `convert_all_to_images`) is forbidden.

### 4.2 S2 — Quality control (`data/qc.py`)

Compute exactly these 8 features per frame, in this order:

```python
FEATURES = ("sector_coverage", "nonblack_ratio", "intensity_mean", "intensity_std",
            "shannon_entropy", "skewness", "kurtosis", "edge_density")
```

- `sector_coverage`: fraction of image area occupied by the ultrasound wedge (from §4.3 sector detection).
- `nonblack_ratio`: fraction of strictly positive grayscale pixels.
- `shannon_entropy`: base-2, over the 256-bin intensity histogram.
- `skewness`, `kurtosis`: `scipy.stats`, Fisher definition for kurtosis (document the choice).
- `edge_density`: mean of a Canny edge map; fix `(low, high)` thresholds in config.

Then z-standardize each feature **across the whole dataset**, set
`qc_flag = any(|z| > 3)`, and write `manifests/qc_review_queue.csv`.

> **Critical:** the flag is a *high-sensitivity screen*, not an exclusion. `qc_excluded`
> is set only by ingesting a human-reviewed CSV (`scripts/01_run_qc.py --apply-review`).
> Frames flagged but not confirmed are retained (paper lines 159–165).

### 4.3 S3 — Preprocessing (`registration.py`, `masking.py`, `preprocess.py`)

```python
def detect_sector(img: np.ndarray) -> Sector: ...            # apex, radii, opening angle
def fit_circular_model(sector: Sector) -> CircleFit: ...
def align_to_reference(img, fit, ref: CircleFit) -> np.ndarray: ...   # translation + rotation + affine
def sector_mask(img, fit) -> np.ndarray: ...                 # zero outside the wedge
def preprocess_image(img) -> np.ndarray: ...                 # → uint8, 112×112, single channel
```

Order: detect → fit → align → mask → resize(112) → center-crop(112).
Masking removes text overlays, ECG traces, device annotations. The reference **circle fit**
is computed once over the training split and cached to `results/reference_sector.json`;
val/test/synthetic all align to that cached reference.

### 4.4 S4 — View classification (`views/`)

Two supported modes, selected by config:

- `mode: keras` — separate conda env from `requirements-view.txt` (TF ≥ 2.13; port
  `predict_generator` → `model.predict`). Invoked as a subprocess, writes `view_pred`/`view_conf`
  back into the master manifest.
- `mode: onnx` — one-time `export_onnx.py` converts the `.h5` to ONNX; inference then runs
  under `onnxruntime` in the main env. **Prefer this.**

Input contract is fixed by the pretrained weights: **224×224 RGB**, so the 112×112 grayscale
preprocessed image is upscaled and channel-replicated *for view classification only*. Never
feed a 224×224 RGB tensor to the diffusion model or `CompactCNN`.

The paper's primary analyses are **all-view**. View labels exist for (a) Figure 1a's
PLAX/A2C/A3C/A4C panel, and (b) optional view-stratified secondary analyses. Do not gate the
primary endpoint on view classification.

### 4.5 S5 — Class-conditional diffusion (`generative/`)

**`unet_cfg.py`** — U-Net with a learned class embedding and CFG label dropout.

```python
class ClassConditionalUnet(nn.Module):
    def __init__(self, dim=64, dim_mults=(1,2,4,8), channels=1,
                 num_classes: int, cond_drop_prob: float = 0.1,
                 resnet_block_groups=8, learned_sinusoidal_dim=16,
                 attn_dim_head=32, attn_heads=4,
                 full_attn=(False, False, False, True)): ...
    def forward(self, x, time, classes, cond_drop_prob=None): ...
    def forward_with_cond_scale(self, x, time, classes, cond_scale=1.0): ...
```

`forward_with_cond_scale` implements paper eq. line 192:
`ε̃ = ε(x,t,∅) + s·[ε(x,t,y) − ε(x,t,∅)]`, `s ≥ 1`.

Attention is applied at higher resolutions only (paper line 188 / ref. [31]).

**`diffusion.py`** — `GaussianDiffusion` supporting conditional sampling.

Fixed hyperparameters from the paper:

| parameter | value | source |
|---|---|---|
| `timesteps` (train) | 1,000 | line 195 |
| `sampling_timesteps` | **500** | line 196 |
| `beta_schedule` | `sigmoid` | line 184 |
| `objective` | `pred_noise` | line 186 (ε-prediction) |
| `image_size` | 112 | line 456 |
| `channels` | 1 (grayscale) | preprocessing |
| `batch_size` | 16 | line 193 |
| `lr` | 8e-5 | line 194 |
| `ema` | on | line 194 |
| `cond_drop_prob` | 0.1 (tunable) | line 190 |
| `cond_scale` `s` | ≥ 1, sweep {1.0, 1.5, 2.0, 3.0} | line 192 |

`sampling_timesteps (500) < timesteps (1000)` ⇒ DDIM path. State this in the README;
it is a deviation-sensitive detail, so `sample.py` must log which sampler ran.

**`dataset.py`**

```python
class LabeledEchoDataset(Dataset):
    """Reads a manifest, returns (image_tensor, class_idx). Train split only."""
    def __init__(self, manifest: Path, split: str, label_col: str,
                 class_map: dict[str, int], image_size: int = 112,
                 augment_horizontal_flip: bool = True): ...
```

**`trainer.py`** — accelerate-based, EMA, AMP, checkpoint every `save_and_sample_every`,
class-balanced preview grids, optional Weights & Biases. FID is optional and off by default
(the paper does not report FID; pooled-histogram divergences are the reported fidelity metric).

**`sample.py`**

```python
def generate(ckpt: Path, n_per_class: dict[str, int], cond_scale: float,
             sampling_steps: int, out_dir: Path, seed: int,
             manifest_out: Path) -> pd.DataFrame: ...
```

> **Leakage guard (non-negotiable, paper line 179):** the generator is fit on the *training
> partition only*. `trainer.py` must call `assert_manifest_invariants` and raise if any
> `val`/`test` row reaches the dataloader. Add an integration test that plants a `test` row
> and asserts the trainer raises.

Generator instances required: `duke_ca` (3 classes: CA/LVH/GEN, per-task retrainable),
`echonet_dynamic_lvef` (3 classes), `echonet_lvh_wt` (2 classes).

### 4.6 S6 — Classifier (`classify/`)

**`models.py` — `CompactCNN`** (paper lines 205–207):

```
[Conv(k=3) → ReLU → MaxPool(2)] × 4  →  AdaptiveAvgPool2d(1)  →  Dropout(p)  →  Linear(→2)
```

Channel progression `1→32→64→128→256` (config-driven; record the exact widths in
`configs/classifier/compactcnn.yaml`). Optimizer Adam + `CosineAnnealingLR`.

**`transforms.py`** — train-time augmentation exactly as line 208–210: random horizontal
flip, rotation up to ±10°, random brightness and contrast jitter. **Val/test: no augmentation.**

**`engine.py`** — train/eval loop. Model selection on **validation encounter-level AUROC**.
Early stopping with patience. The test partition is touched exactly once, at the end.

**`two_step.py`** (paper lines 221–226):

```python
@dataclass
class TwoStepResult:
    p_abnormal: np.ndarray          # Stage 1: (CA ∪ LVH) vs GEN
    p_ca_given_abnormal: np.ndarray # Stage 2: CA vs LVH
    p_ca: np.ndarray
```

Default composition `p_ca = p_abnormal * p_ca_given_abnormal`, exposed as
`compose: {product | stage2_only | logit_sum}` in config. **See §10, open question 1.**

Synthetic CA images are injected into the training partition of the relevant stage only.

### 4.7 S7 — Training paradigms (`classify/paradigms.py`)

| paradigm | train set | test set |
|---|---|---|
| `TRTR` | real train | real test |
| `TSTR` | synthetic only | real test |
| `TRSTR` | real train + synthetic | real test |
| `BOOTSTRAP` | real train + resampled real minority | real test |

**Synthetic-count selection** (lines 214–219): grid over `{100, 200, 300, …, N_max}`, train on
train, score on **validation** AUROC, take the argmax, evaluate **once** on test. The test
partition selects nothing — not the count, not the classifier, not the epoch, not any
hyperparameter. Enforce this in code with a `TestSetGuard` context manager that refuses to
serve test predictions more than once per run and logs the access.

**Bootstrap baseline** (lines 245–250): resample real minority-class *training* rows with
replacement; the number added must equal the number of synthetic samples used in the matched
TRSTR run; majority class untouched; a fresh bootstrap sample per training run; all other
settings identical.

### 4.8 S8 — Evaluation (`eval/`)

```python
# aggregate.py
def to_encounter_level(df: pd.DataFrame, prob_col="p", by="encounter_id") -> pd.DataFrame:
    """Mean of image-level probabilities within encounter. This is the ONLY reported unit."""

# metrics.py
def auroc(y, p) -> float
def auprc(y, p) -> float

# delong.py  (Sun & Xu fast DeLong)
def delong_auc_variance(y, p) -> tuple[float, float]        # auc, var
def delong_ci(y, p, alpha=0.05) -> tuple[float, float, float]
def delong_paired_delta_ci(y, p1, p2, alpha=0.05) -> DeltaResult  # ΔAUC, CI, z, p
```

AUPRC is reported as a point estimate only (line 239) — no CI, do not invent one.

### 4.9 S9 — Fidelity (`eval/fidelity.py`, `embedding.py`, `discriminator.py`)

**Pooled grayscale histogram** — reproduce the paper's procedure *exactly*, including its
quirks, because the reported numbers depend on them:

1. Load with **Pillow**, convert to 8-bit grayscale (`"L"`).
2. Accumulate counts into a 256-bin NumPy histogram over `[0, 256)`.
3. **Do not resize and do not weight images equally** — each image contributes in proportion
   to its pixel count (line 273–274).
4. Add `1e-12` to every bin, then normalize.
5. Report `KL(real‖syn)`, `KL(syn‖real)`, Jensen–Shannon **divergence**, and JS **distance**
   (= √divergence) as separate numbers.
6. `Brier_dist = mean_b (p_b − q_b)²`; `RMSE_dist = √Brier_dist`.

**Embeddings** — t-SNE and UMAP, both, on the same feature matrix. These are **qualitative
diagnostics**; the plotting function must not emit any p-value or test statistic.

**Real-vs-synthetic discriminator** — binary CNN, **patient-level** 70/15/15 split, model
selection on val, single evaluation on test, report AUROC with DeLong CI. An AUROC of 0.5
means no separability; the observed value is interpreted alongside its CI and class
composition (lines 286–293).

### 4.10 S10 — Survey analysis (`eval/survey.py`)

- Denominators are **item-wise** (missing responses excluded per item), so every reported
  percentage carries its own `n`.
- Normalized entropy: `H_i = −Σ_k p_ik log₂ p_ik`, `H_norm = H_i / log₂(K)`, `K` = number of
  response categories **for that item**. Averaged within the real and synthetic groups.
- Responses are **not independent** (each evaluator rates multiple images) — the module must
  emit descriptive statistics only and must refuse to produce confirmatory p-values.
- Paired-comparison responses are recoded after unblinding into
  `{prefers_real, prefers_synthetic, cannot_decide}`.
- Output reproduces Table 2 as a tidy CSV plus a formatted markdown table.

### 4.11 S11 — EchoNet adapters (`data/echonet.py`)

- **Dynamic:** decode videos → first frame → same preprocessing → LVEF 3-class bucketing.
- **LVH:** drop videos missing `IVSd` or `LVPWd` (349 → 11,651 retained); `max_wt = max(IVSd, LVPWd)`;
  τ from train only; apply τ unchanged to val and test. Assert the resulting counts:
  train 9,152/1,019, val 1,037/104, test 210/129.
  > Sanity note to surface at runtime: the official test partition is ~38% minority versus
  > ~10% in train. That is a property of the released split, not a bug — but log a loud
  > warning so nobody silently reads AUPRC across partitions as comparable.

---

## 5. Configuration

YAML files composed by a small loader (`utils/config.py`) with `--set key=value` CLI
overrides. Every run writes its **fully resolved** config next to its results. Prefer plain
YAML + dataclass validation over Hydra; the reference repo's argparse-and-wandb-sweep pattern
is preserved for hyperparameter sweeps only.

```yaml
# configs/experiment/duke_two_step_allview.yaml
name: duke_two_step_allview
seed: 2024
data:   {manifest: manifests/duke_master.csv, image_size: 112, views: all}
task:   {type: two_step, compose: product}
classifier: {_base: configs/classifier/compactcnn.yaml, epochs: 100, patience: 15,
             lr: 1.0e-3, batch_size: 32, dropout: 0.3}
generator:  {_base: configs/diffusion/duke_ca.yaml, ckpt: results/runs/<id>/ckpt/ema_best.pt,
             cond_scale: 2.0, sampling_steps: 500}
paradigms: [TRTR, TRSTR, BOOTSTRAP]
synthetic_grid: {start: 100, stop: 3000, step: 100}
eval: {unit: encounter_id, metrics: [auroc, auprc], ci: delong}
```

---

## 6. Experiment matrix

| id | task | dataset | paradigms | status |
|---|---|---|---|---|
| E1 | two-step CA, all views | Duke | TRTR, TRSTR, BOOTSTRAP | **primary endpoint** |
| E2 | CA detection | Duke | TRTR, TSTR | secondary diagnostic |
| E3 | wall-thickness phenotype | EchoNet-LVH | TRTR, TRSTR | secondary |
| E4 | fidelity | EchoNet-Dynamic | — | exploratory |
| E5 | real-vs-synthetic discriminator | Duke | — | exploratory |
| E6 | expert survey | Duke | — | descriptive |
| E7 | view-stratified two-step | Duke | TRTR, TRSTR | optional |

---

## 7. Reproduction targets

`tests/integration/test_reproduction.py` compares against these. On real data, values
should land inside the reported CIs; on the synthetic smoke fixtures they only need to run.

**Primary (Table 3)**

| task | real-only | bootstrap | real+synth | ΔAUROC (95% CI) |
|---|---|---|---|---|
| Duke CA, all views | 0.883 (0.831–0.927) | 0.899 (0.856–0.934) | 0.924 (0.883–0.955) | 0.041 (0.013–0.069) |
| EchoNet-LVH | 0.832 (0.784–0.874) | — | 0.864 (0.824–0.901) | 0.032 (0.021–0.044) |

**Secondary TRTR/TSTR:** TRTR AUROC 0.906 (0.859–0.941), AUPRC 0.613; TSTR AUROC 0.889
(0.842–0.936), AUPRC 0.575; ΔAUROC −0.017 (−0.075–0.036).

**Discriminator:** train 0.907 (0.887–0.925), val 0.885 (0.853–0.914), test 0.612 (0.537–0.692).

**Fidelity:** KL(real‖syn) 0.019, KL(syn‖real) 0.017, JSD 0.0043, JS distance 0.066,
Brier 8.8e-6, RMSE 0.0030. *(Internal check: 0.066² = 0.00436 ≈ JSD, and √8.8e-6 = 0.00297 ≈ RMSE
— both consistent, so a correct implementation should hit these jointly.)*

**Survey (Table 2), synthetic vs real:** ROI adequate 19.2% / 57.3%; clinically usable
27.5% / 70.9%; strictly diagnostic 3.6% / 31.7%; not confidently identified as synthetic
65.3% / 87.9%; mean identifiable structures 1.1 / 2.2. Paired realism across 85 responses:
40.0% cannot decide, 31.8% real, 28.2% synthetic.

---

## 8. Testing

**8.1 Unit** — QC feature correctness on synthetic patterns; sector detection on a drawn
wedge; masking idempotence; DeLong against a reference implementation on a fixed vector;
histogram divergences on hand-computed distributions (KL of identical distributions = 0);
entropy normalization (`K` uniform ⇒ 1.0); manifest invariants.

**8.2 Integration (smoke)** — a `tests/fixtures/tiny/` set of ~60 procedurally generated
112×112 wedge images across 3 classes and 20 encounters. End-to-end: split → 200-step
diffusion train → sample 10/class → CompactCNN 3 epochs → encounter-level AUROC → report.
Must finish in **< 5 minutes on CPU**. This is the agent's primary feedback loop.

**8.3 Leakage tests** — (a) a `test` row planted in the generator dataloader raises;
(b) synthetic rows in val/test raise; (c) `TestSetGuard` raises on a second access;
(d) an encounter split across partitions raises.

**8.4 Repo hygiene** — pre-commit hook rejecting `*.dcm`, `*.png`, `*.pt`, `*.h5`, `data/**`,
and any file matching a PHI-ish pattern (MRN-like digit runs, dates of birth).

---

## 9. HPC

SLURM templates in `shell/slurm/`, following the reference repo's pattern but parameterized:

- `train_diffusion.sh` — 1 GPU (A5000/A100), 50–80 GB RAM, long walltime, `logs/%j.out`.
- `sample.sh` — 1 GPU, short.
- `train_classifier.sh` — 1 GPU, short; array job over the synthetic-count grid
  (`#SBATCH --array=0-29` for 100…3000 by 100).
- `preprocess.sh` — CPU-only array over DICOM shards.

No email addresses, partitions, or paths hard-coded — read them from
`shell/slurm/cluster.env` (gitignored, with a committed `.example`).

---

## 10. Open decisions to confirm before the agent starts

These are genuinely underdetermined by the manuscript. Pick an answer, write it into
`configs/`, and record it in `docs/DECISIONS.md` so the methods section and the code agree.

1. **Two-step composition.** The paper describes the two stages but not how their outputs
   combine into the single encounter-level CA score behind AUROC 0.924. `product` is the
   natural reading; `stage2_only` (evaluating CA vs LVH among predicted-abnormal) is the
   other plausible reading and gives different numbers. **This must be nailed down.**
2. **Averaging order.** Image-level probabilities are averaged within encounter (line 244) —
   but for a two-step model, do you average each stage's probability then compose, or compose
   per image then average? Default: compose per image, then average.
3. **Diffusion channels.** Preprocessed images are grayscale, but the reference repo defaults
   to `channels=3`. Default to 1 and document it.
4. **Generator scope for `duke_ca`.** One 3-class conditional model over {CA, LVH, GEN}, or a
   CA-only model? Line 176–177 implies one task-specific conditional model per experiment;
   default to the 3-class model and sample only the CA class for augmentation.
5. **`N_max` for the synthetic grid.** Not stated. Default `stop: 3000` (≈14× the 216 real
   training CA encounters) and report the selected value.
6. **Sector-registration reference.** Which training image (or mean geometry) defines the
   reference coordinate system. Default: median fitted circle over the training split.
