# Reviewer revision analysis map

## Released code versus protected results

This release organizes source and configuration, not protected clinical outputs.
No claims of new training or improved performance follow from a code release.
The root package remains the general reference implementation; the distinct
full-archive source is in `workflows/patient_sensitivity`.

| Reviewer request | Public entry point | Scope |
|---|---|---|
| Patient-independent split, aggregation | patient_sensitivity source and configs | Source reconstruction; governed data required |
| Five classifier seeds | patient_sensitivity README seed loop | Shared generator; not five generator seeds |
| Calibration | analyses/followup/01_calibration.py | Fixed ensemble, patient bootstrap |
| Era/vendor | analyses/acquisition/era_vendor_analysis.py | Patient strata, acquisition-only and exploratory OOF models |
| View/phase availability | analyses/followup/03_view_phase_audit.py | Availability audit is not validated view/phase inference |
| No-flip | analyses/followup/05_run_no_flip.py | One-seed exploratory rerun |
| Operating point | analyses/operating_point/final_operating_point.py | Threshold selected on validation, applied to real test patients |
| Privacy and diversity | analyses/followup/04_privacy_nearest_neighbor.py | Normalized-pixel audit; not a privacy guarantee |
| GAN / structural fidelity | Not reported in this release | Institutional jobs pending; no completion claim |

Paths under `analyses/` above are relative to `workflows/patient_sensitivity`.
Read each CLI's `--help` and the runbook before running. Outputs can include
protected patient-level tables and must remain in the governed environment.

## Image-count interpretation

The existing cap5 comparison retrained a real-only classifier and capped both
training and evaluation images. A lower AUROC than uncapped real-only training
shows sensitivity to available images/selection, not that synthetic augmentation
solves the problem. It is not an inference-only comparison.

A future matched experiment would use identical capped real manifests for TRTR
and TRSTR, preserve patient partitions, restrict generator access to training
patients, select synthetic count only on capped validation data, and evaluate the
same capped real test patients. Report paired patient-level AUROC/AUPRC differences
and per-seed results. To test whether augmentation helps *more* under capping,
compare its paired gain with a matched uncapped gain; a positive capped gain alone
does not establish an interaction. This experiment has not been performed by
this release and has no reported result.

## Release validation

- Generic fast unit suite: 44 passed, 2 deselected (slow tests excluded).
- Separate full-archive mock smoke: PASS on 42 artificial DICOM patients/images,
  zero patient overlap, including TRTR, TRSTR and comparator execution.
- Validation ran on a local CPU with PyTorch 2.5.1, not the institutional CUDA
  environment. It checks pipeline execution, not clinical performance reproduction.
- No protected cohort or GPU training was used for release validation.
