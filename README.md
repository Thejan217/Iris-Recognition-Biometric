# Iris Recognition Pipeline & Presentation Attack Analysis

A biometric authentication pipeline for iris recognition, built on the
MMU Iris Database using MobileNetV2 transfer learning. Developed as
my individual contribution to a team biometrics assignment, where
each member built a separate pipeline for a different modality
(fingerprint, iris, face, vein).

## What it does

- Loads and preprocesses the MMU Iris Database (45 subjects, 90
  eye-classes, 450 near-infrared images), with CLAHE applied to
  handle low-contrast iris images
- Fine-tunes a MobileNetV2 backbone (pretrained on ImageNet) with a
  trainable 128-dimensional embedding head, producing L2-normalised
  iris templates
- Builds master templates by averaging each eye's training
  embeddings, then verifies held-out probe images against them via
  cosine similarity
- Evaluates with standard biometric metrics: EER, AUC, FMR, FNMR, ATV
- Simulates four presentation attacks to test spoofing resistance
- Estimates Failure-to-Enrol Rate using an image-quality proxy
  (sharpness and contrast)

## Why transfer learning

With only four training images per eye, this dataset provides three
orders of magnitude fewer samples than typically required to train a
deep network from random initialisation. ImageNet's pretrained
low-level filters give a usable starting point where training from
scratch wouldn't.

## Results

| Metric | Value |
|--------|-------|
| AUC | 0.998 |
| EER | 2.01% |
| FMR | 1.8% |
| FNMR | 2.22% |
| ATV | 98.2% |

Verification performance is competitive with reported small-dataset
iris results.

### Presentation attack resistance

Four simulated attacks against the live probe set:

| Attack | ATSR (success rate) |
|--------|---------------------|
| Blur | 0.0% |
| Low resolution | 1.1% |
| Gaussian noise | 10.0% |
| **Screen glare** | **85.6%** |

The system is robust to blur, low resolution and noise, but a
uniform-brightness attack simulating a phone screen or printed photo
succeeds 85.6% of the time. This indicates pretrained convolutional
features preserve relative texture patterns even under significant
illumination shifts — the model is matching texture structure, not
confirming it's looking at a live eye. Any real deployment would need
dedicated hardware liveness detection; this vulnerability is the main
finding of the practical work, not an afterthought.

### Enrolment quality

Failure-to-Enrol was estimated via a sharpness/contrast quality
proxy, since the curated MMU dataset doesn't exercise genuine
sensor-level quality checks: **5.11% proxy FTER** (23 of 450 images
flagged as low-quality).

## Legal and compliance context

As part of the wider team assignment, I authored the compliance
analysis assessing biometric data as a special category of personal
data under UK GDPR (Article 9), including why consent is difficult
to establish as a lawful basis in a border-control-style capture
scenario given the power imbalance between data subject and
controller (GDPR Recital 43).

## Scope note

This was an individual pipeline within a four-person team assignment.
I selected the iris modality, independently sourced the dataset, and
built, trained and evaluated this pipeline myself. The team's shared
output was a comparative recommendation across all four modalities
for an airport immigration use case.

## Dataset

[MMU Iris Database](http://pesona.mmu.edu.my/~ccteo/) — not included
in this repo; subject to its own usage terms, source separately.

## Stack

Python · TensorFlow/Keras · MobileNetV2 · OpenCV · scikit-learn
