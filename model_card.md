# Model Card: Histology VAE Student

This model card follows the spirit of Mitchell et al. (2019),
"Model Cards for Model Reporting," adapted for a representation-
learning research project rather than a deployed classifier.

## Model details

- **Models**: deterministic convolutional autoencoder, convolutional
  VAE, beta-VAE, in four training variants (RGB, grayscale, RGB
  color-denoising, and a non-learned RGB/HSV color-statistics
  baseline).
- **Architecture**: convolutional encoder/decoder, 4 downsampling
  blocks, 32–64 dimensional latent space (configurable via
  `configs/*.yaml`).
- **Framework**: PyTorch (CPU-only checkpoints by default; GPU
  supported via `device.accelerator: cuda`).
- **Developed by**: an individual applicant, as a learning/portfolio
  project. This is not a BostonGene product and was not developed
  with proprietary or patient data.
- **Version**: pilot subset, single training seed per model
  (see `docs/RESULTS.md` for exact configuration).

## Intended use

- **In scope**: studying unsupervised representation learning and
  color/stain shortcuts on public colorectal H&E patches; comparing
  representation quality via linear probing and nearest-neighbor
  retrieval; as a teaching example of a reproducible ML research
  pipeline.
- **Out of scope**: clinical decision-making, diagnosis, triage, or
  any use involving real patient data. The models were never trained
  or validated for that purpose and no such validation is claimed.

## Training data

- **NCT-CRC-HE-100K** (training pool) and **CRC-VAL-HE-7K** (external
  test), both public colorectal histology patch datasets, 9 tissue
  classes (`ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`,
  `TUM`).
- Pilot subsets only: 4,860 train / 540 validation / 1,800 external
  test images. Internal train/validation split is patch-level, not
  patient-level, because the public archive does not expose reliable
  patient identifiers.
- Class labels are used only for post-hoc evaluation, never for VAE
  training.
- A small synthetic patch generator (`src/datasets/synthetic.py`) is
  used for unit tests and CI, and produces no biologically meaningful
  images.

## Evaluation data and metrics

- External linear probe (logistic regression on frozen embeddings,
  regularization selected on validation, refit on train+validation,
  scored on the held-out external test set): balanced accuracy,
  macro-F1, log loss, confusion matrices.
- Paired stratified bootstrap (2,000 iterations) for model-vs-model
  comparisons on the external test set.
- Cosine/Euclidean nearest-neighbor retrieval (top-1, MRR,
  precision@k, hit rate@k).
- Full numeric results: `docs/RESULTS.md`.

## Quantitative results (external test, 32-dim representation)

| Representation | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| RGB VAE | 0.3478 | 0.3256 |
| Color-denoising VAE | 0.3961 | 0.3555 |
| Grayscale VAE | 0.4122 | 0.3872 |
| RGB-HSV color PCA (non-learned) | 0.6528 | 0.6490 |

Balanced random-class reference: ~0.1111.

## Known limitations and failure modes

- **Color shortcut**: a non-learned color-statistics baseline
  substantially outperforms every VAE variant, indicating that stain
  and brightness statistics — not tissue architecture — explain most
  of the linearly separable signal in this dataset.
- **Single seed**: each main model was trained with one random seed;
  reported numbers do not include across-seed variance.
- **Patch-level, not patient-level**: internal validation and
  bootstrap intervals are computed at the patch level, because
  patient identifiers are not reliably available in the public
  training archive. This is a data limitation, not a tooling gap:
  `src/analysis/patient_level.py` (leakage checks, majority-vote
  patient metrics, and a patient-cluster bootstrap) already exists,
  is unit-tested, and is demonstrated end to end on synthetic data in
  `notebooks/05_patient_level_analysis.ipynb`. That notebook also
  shows that patch-level bootstrap CIs, as used elsewhere in this
  project, should be read as a **lower bound** on true uncertainty
  when patch errors are correlated within a patient.
- **Small convolutional architecture**: capacity and receptive field
  are modest; results should not be extrapolated to larger
  architectures or full-resolution whole-slide images without
  re-validation.
- **No segmentation ground truth**: nuclei-segmentation features
  (`src/analysis/nuclei_segmentation.py`) are produced by a classical
  Otsu + watershed pipeline with no pixel-level annotation to
  validate against; treat nucleus counts/shape statistics as
  approximate, exploratory features, not validated measurements.
- **Frozen pretrained baseline is out-of-domain**: the optional
  DINOv2 baseline (`src/models/pretrained_encoder.py`) is pretrained
  on natural images, not histology, and is not fine-tuned here.
- **No multimodal or generative extension**: this project does not
  include a diffusion/DDPM model, RNA-seq data, or multiplexed
  immunofluorescence; see `README.md#roadmap` for how those would be
  approached.

## Ethical considerations and data governance notes

- Only public, de-identified patch datasets were used. No patient
  outcomes, treatment data, or proprietary/confidential information
  are present anywhere in this repository.
- The project does not make or imply any clinical claim. Any reuse of
  these models for real diagnostic support would require, at minimum:
  patient-level (not patch-level) evaluation, multi-seed and
  multi-institution validation, prospective clinical validation, and
  review under applicable regulatory and Good Clinical/Laboratory
  Practice (GCP/GCLP) frameworks — none of which has been done here.
- If extended to real patient data in the future, this project would
  need: an auditable data-lineage record from raw slide to reported
  metric, access controls, and a documented data retention/deletion
  policy, in line with GCP/GCLP-style record keeping. These controls
  are out of scope for this public, portfolio-stage repository.

## LLM-assisted components

`src/reporting/llm_report.py` optionally uses the Anthropic API to
draft natural-language summaries **from already-computed
`metrics.json` files**. The LLM never sees raw images or produces
metrics itself; generated text is checked against the source numbers
(see `find_unverified_numbers`) and any unverified figure is flagged
before a human reads the draft. This is a drafting aid, not an
evaluation method.
