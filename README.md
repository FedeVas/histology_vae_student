# Unsupervised Histology Representation Learning with Variational Autoencoders

A reproducible PyTorch project for studying unsupervised representation
learning and shortcut effects in colorectal H&E histology patches.

The project compares standard RGB variational autoencoders, grayscale
training, color-denoising training and a spatially invariant color-only
baseline.

## Project motivation

Histology models may achieve high classification performance by using
stain, brightness or acquisition-related shortcuts rather than tissue
architecture.

This project asks:

> How much transferable tissue information is learned by a baseline VAE,
> and how much of the apparent class signal can be explained by color
> alone?

Labels are not used to train the autoencoders. They are used only for
post-hoc representation evaluation.

## Public data

The experiments use public colorectal histology patches from:

- NCT-CRC-HE-100K;
- CRC-VAL-HE-7K.

The pilot contains nine tissue classes:

`ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`, and `TUM`.

| Split | Images |
|---|---:|
| Train | 4,860 |
| Validation | 540 |
| External test | 1,800 |

The internal validation split is patch-level because reliable
patch-to-patient mapping is unavailable in the public training archive.
The external test set comes from a separate public dataset.

## Models and controls

The project implements:

- deterministic convolutional autoencoder;
- convolutional VAE;
- beta-VAE support;
- RGB VAE;
- grayscale VAE;
- RGB color-denoising VAE;
- RGB/HSV spatially invariant color baseline;
- PCA capacity matching;
- linear probing;
- paired bootstrap comparison;
- nearest-neighbor morphology retrieval;
- Euclidean and cosine retrieval sensitivity analysis.

## Main results

### External linear probe

| Representation | Dimensions | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| RGB VAE | 32 | 0.3478 | 0.3256 |
| Color-denoising VAE | 32 | 0.3961 | 0.3555 |
| Grayscale VAE | 32 | **0.4122** | **0.3872** |
| RGB-HSV PCA | 32 | **0.6528** | **0.6490** |

The balanced random-class reference is approximately `0.1111`.

The color-only representation is substantially stronger than every VAE,
showing that color is a major shortcut in this dataset.

Among the VAE variants, grayscale preprocessing provides the strongest
overall external performance.

### Cosine nearest-neighbor retrieval

| Model | Top-1 | MRR | Precision@5 | Hit rate@5 |
|---|---:|---:|---:|---:|
| Grayscale VAE | **0.3539** | **0.4824** | 0.3313 | **0.7172** |
| Color-denoising VAE | 0.3494 | 0.4760 | **0.3352** | 0.6939 |

The learned latent spaces contain non-random local tissue structure, but
complex epithelial, stromal and inflammatory classes remain mixed.

Detailed results, bootstrap intervals and retrieval examples are
available in [docs/RESULTS.md](docs/RESULTS.md).

## Key conclusion

The experiments demonstrate that strong downstream classification does
not necessarily imply strong morphological representation learning.

Global color statistics provide the strongest class signal, while
removing or perturbing color improves the transferability of VAE latent
representations relative to a standard RGB reconstruction objective.

## Repository structure

```text
configs/
    Experiment configuration files.

data/
    Local datasets and generated metadata.
    Raw image data is not tracked by Git.

docs/
    Final results and selected figures.

src/
    datasets/
        Metadata, transformations and dataset factories.
    models/
        Autoencoder, VAE and frozen pretrained-encoder (DINOv2)
        implementations.
    training/
        Training engine, losses and checkpoints.
    analysis/
        Reconstruction, latent, probe, color, nuclei-segmentation
        and retrieval analyses.
    reporting/
        Optional LLM-assisted experiment-summary drafting.

tests/
    Unit and integration tests.

outputs/
    Local training runs and analysis outputs.
    Large outputs are not tracked by Git.
```

## Installation

Create a virtual environment and install the project dependencies.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the dependencies recorded for the project:

```bash
python -m pip install -r requirements.txt
```

The default `torch`/`torchvision` entries install CPU wheels. If you
have a CUDA GPU, install a matching build from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
instead, then run the rest of `requirements.txt`.

### Run tests

```bash
python -m pytest -q
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the
same test suite, plus a CPU smoke-test training run on synthetic
data, on every push and pull request.

### Train a model

Standard RGB VAE:

```bash
python -m src.train --config configs/crc_vae_pilot_cpu.yaml
```

Grayscale VAE:

```bash
python -m src.train --config configs/crc_grayscale_vae_pilot_cpu.yaml
```

Color-denoising VAE:

```bash
python -m src.train --config configs/crc_color_denoising_vae_pilot_cpu.yaml
```

A short engineering check can be run with:

```bash
python -m src.train --config configs/crc_vae_pilot_cpu.yaml --smoke-test
```

### Extract embeddings

```bash
python -m src.evaluate --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --split train
python -m src.evaluate --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --split validation
python -m src.evaluate --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --split test
```

Evaluation loaders are deterministic and do not use random training
augmentations.

### Linear probe

```bash
python -m src.run_linear_probe --train-embeddings TRAIN_CSV --validation-embeddings VALIDATION_CSV --test-embeddings TEST_CSV --feature-prefix latent_ --output-dir OUTPUT_DIRECTORY
```

For color features:

```bash
python -m src.run_linear_probe --train-embeddings TRAIN_COLOR_CSV --validation-embeddings VALIDATION_COLOR_CSV --test-embeddings TEST_COLOR_CSV --feature-prefix color_ --pca-components 32 --output-dir OUTPUT_DIRECTORY
```

### Nearest-neighbor retrieval

```bash
python -m src.run_retrieval --train-embeddings TRAIN_CSV --query-embeddings TEST_CSV --output-dir OUTPUT_DIRECTORY --k-values 1 3 5 --metric cosine --queries-per-class 3 --montage-neighbors 5
```

Labels are not used to identify neighbors. They are applied only after
retrieval to calculate class agreement.

## Extended experiments

These additions sit alongside the core VAE study and share its
reproducibility discipline (deterministic evaluation, train-only
fitting, external test set), but are not yet part of the main
`docs/RESULTS.md` comparison.

### Classical nuclei segmentation (OpenCV + scikit-image)

`src/analysis/nuclei_segmentation.py` adds a non-learned nuclei
segmentation baseline: hematoxylin color deconvolution, Otsu
thresholding, watershed splitting of touching nuclei, and
region-based shape/count features (nucleus count, density,
eccentricity, solidity). This targets the "image segmentation" and
"OpenCV / scikit-image" parts of a typical digital-pathology CV role
that the VAE study alone does not cover.

```bash
python -m src.extract_segmentation_features --metadata METADATA_CSV --output-dir OUTPUT_DIRECTORY
```

The resulting `segmentation_*` columns can be fed into the same
`src.run_linear_probe` pipeline as the VAE and color features
(`--feature-prefix segmentation_`).

### Frozen self-supervised encoder baseline (DINOv2)

`src/models/pretrained_encoder.py` wraps a frozen, ImageNet-pretrained
DINOv2 checkpoint (no histology fine-tuning) so it can be compared,
on the same external test set and the same linear-probe/retrieval
pipeline, against the VAE variants and the color-shortcut baseline.

```bash
python -m src.extract_pretrained_embeddings --metadata METADATA_CSV --split test --encoder-name dinov2_vits14 --output-dir OUTPUT_DIRECTORY
```

Requires internet access on first use (weights are fetched via
`torch.hub`).

### LLM-assisted experiment reporting

`src/reporting/llm_report.py` optionally uses the Anthropic API to
turn one or more `metrics.json` files (already produced by
`src/evaluate.py`, `src/run_linear_probe.py`, `src/run_retrieval.py`)
into a draft markdown summary. The LLM never computes a metric
itself; it only narrates numbers it is given, and the generated text
is checked against those numbers before being written to disk.

```bash
pip install -r requirements-llm.txt
export ANTHROPIC_API_KEY=...
python -m src.generate_llm_report --metrics rgb_vae=outputs/rgb_vae/probe/metrics.json --metrics grayscale_vae=outputs/grayscale_vae/probe/metrics.json --output docs/AUTO_SUMMARY.md
```

### Unsupervised clustering

`src/analysis/clustering.py` fits KMeans/Gaussian-mixture clustering
on **train** embeddings only (no labels used) and evaluates
agreement with true tissue classes (purity, adjusted Rand index,
normalized mutual information) on held-out data — the same
train-only-fit / external-eval discipline as the linear probe.

```bash
python -m src.run_clustering --train-embeddings TRAIN_CSV --validation-embeddings VALIDATION_CSV --test-embeddings TEST_CSV --feature-prefix latent_ --number-of-clusters 9 --output-dir OUTPUT_DIRECTORY
```

### Patient-level analysis tooling

`src/analysis/patient_level.py` provides patient-aware evaluation:
an automated leakage check between splits, majority-vote
patient-level metrics, and — importantly — a **cluster bootstrap**
that resamples patients rather than patches. The patch-level
bootstrap used elsewhere in this project (and in `docs/RESULTS.md`)
treats each patch as independent; when a patient's patches share
correlated errors, that understates true uncertainty. This module
exists and is tested now (see `notebooks/05_patient_level_analysis.ipynb`
for a worked, synthetic-data demonstration) so it is ready to apply
the moment a patient-mapped cohort is available — see "Roadmap"
below for what that would take for the real CRC data.

### Interactive notebooks

`notebooks/01_data_qc.ipynb` through `05_patient_level_analysis.ipynb`
cover data QC, reconstruction inspection, latent-space projection
(PCA/UMAP), clustering, and patient-level bootstrap analysis. Each
notebook prefers real pipeline outputs (embeddings, checkpoints) if
present, and otherwise falls back to the synthetic dataset and the
classical (non-learned) feature extractors already in the project,
so every notebook runs end to end without training anything first.

## Reproducibility

The project uses:

- configuration-driven experiments;
- fixed random seeds;
- deterministic evaluation transforms;
- train-only fitting for preprocessing and model selection;
- external dataset evaluation;
- saved checkpoints and machine-readable metrics;
- automated tests (see `.github/workflows/tests.yml` for CI);
- explicit shortcut controls.

## Scope and limitations

This project evaluates representation quality on public tissue-class
patches. It does not make clinical predictions and does not use patient
outcomes, treatment response, proprietary datasets or confidential
patient information.

Results are based on pilot subsets and one main training seed per model.
Internal validation and bootstrap analyses are patch-level.

See `model_card.md` for a fuller account of intended use, known
failure modes and data-governance notes.

## Roadmap

Directions this project would extend into for broader, multimodal
digital-pathology work:

- **RNA-seq**: a parallel single-modality encoder (e.g. a
  count-based VAE in the style of scVI) trained on a public bulk or
  single-cell colorectal cancer RNA-seq cohort (e.g. TCGA-COAD/READ),
  evaluated with the same train/validation/external-test discipline
  used here for images.
- **Multiplexed immunofluorescence (mIF)**: extending the patch
  pipeline in `src/datasets` to multi-channel, multi-marker inputs
  (e.g. a public CODEX or Orion mIF dataset), with per-channel
  normalization replacing the single-channel color-shortcut controls
  used for H&E.
- **Diffusion / DDPM**: an unconditional or class-conditional DDPM
  over the same patch distribution as the VAEs, compared on sample
  quality (FID-style metrics) and, if paired with RNA-seq, as a first
  step toward a cross-modal (image ↔ expression) generative model.
- **Multimodal fusion**: combining the image encoder here with an
  RNA-seq encoder through a shared latent space or cross-attention,
  evaluated on whether the joint representation is more
  transferable than either modality alone — directly testing whether
  the color-shortcut problem found here persists or is diluted once
  a second modality is available.

None of the above is implemented in this repository yet; this section
exists so scope and next steps are explicit rather than implied.

## License

Project source code is released under the MIT License (see
`LICENSE`). Public datasets referenced by this project
(NCT-CRC-HE-100K, CRC-VAL-HE-7K) remain subject to their own original
licenses and usage terms.
