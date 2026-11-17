# EchoSynth model card

## Intended use

EchoSynth is a research pipeline for class-conditional generation of single-frame grayscale
echocardiograms and for evaluating whether training-only synthetic augmentation changes
classification performance on held-out real patients. It is not a medical device and is not
intended for diagnosis, triage, treatment selection, or autonomous clinical decision-making.

## Evaluation unit and leakage control

All encounters from a unique patient are assigned to one train, validation, or test partition.
Synthetic images are used only in the training partition. Image probabilities are averaged within
encounter and then across encounters within patient; reported downstream metrics use patients as
the independent evaluation unit. The validation partition selects checkpoints and synthetic
counts before the test partition is accessed.

## Inputs and outputs

The diffusion model consumes a 112 by 112 single-channel image, a diffusion timestep, and a class
label. The downstream classifier consumes a 112 by 112 single-channel image and outputs class
probabilities. The reference configuration uses 1,000 forward diffusion steps and 100-step DDIM
sampling with classifier-free guidance scale 2.0.

## Limitations

The system generates single frames rather than temporally coherent videos. View, cardiac phase,
acquisition year, and vendor are not standardized conditioning variables. Synthetic images may
contain artifacts and must not be treated as clinically interchangeable with real studies.
Performance depends on local cohort composition and acquisition practice. A negative result from
one nearest-neighbour or membership audit does not guarantee privacy against other attacks.

## Data and release policy

Duke images, protected identifiers, manifests, model weights, and institutional run artifacts are
not distributed in this repository. Release of models trained on protected data requires
institutional review and an appropriate data-governance process.
