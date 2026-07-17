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
        Autoencoder and VAE implementations.
    training/
        Training engine, losses and checkpoints.
    analysis/
        Reconstruction, latent, probe, color and retrieval analyses.

tests/
    Unit and integration tests.

outputs/
    Local training runs and analysis outputs.
    Large outputs are not tracked by Git.

Installation

Create a virtual environment and install the project dependencies.

python -m venv .venv

Windows:

.venv\Scripts\activate

Install the dependencies recorded for the project:

python -m pip install -r requirements.txt
Run tests
python -m pytest -q
Train a model

Standard RGB VAE:

python -m src.train --config configs/crc_vae_pilot_cpu.yaml

Grayscale VAE:

python -m src.train --config configs/crc_grayscale_vae_pilot_cpu.yaml

Color-denoising VAE:

python -m src.train --config configs/crc_color_denoising_vae_pilot_cpu.yaml

A short engineering check can be run with:

python -m src.train --config configs/crc_vae_pilot_cpu.yaml --smoke-test
Extract embeddings
python -m src.evaluate --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --split train
python -m src.evaluate --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --split validation
python -m src.evaluate --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --split test

Evaluation loaders are deterministic and do not use random training
augmentations.

Linear probe
python -m src.run_linear_probe --train-embeddings TRAIN_CSV --validation-embeddings VALIDATION_CSV --test-embeddings TEST_CSV --feature-prefix latent_ --output-dir OUTPUT_DIRECTORY

For color features:

python -m src.run_linear_probe --train-embeddings TRAIN_COLOR_CSV --validation-embeddings VALIDATION_COLOR_CSV --test-embeddings TEST_COLOR_CSV --feature-prefix color_ --pca-components 32 --output-dir OUTPUT_DIRECTORY
Nearest-neighbor retrieval
python -m src.run_retrieval --train-embeddings TRAIN_CSV --query-embeddings TEST_CSV --output-dir OUTPUT_DIRECTORY --k-values 1 3 5 --metric cosine --queries-per-class 3 --montage-neighbors 5

Labels are not used to identify neighbors. They are applied only after
retrieval to calculate class agreement.

Reproducibility

The project uses:

configuration-driven experiments;
fixed random seeds;
deterministic evaluation transforms;
train-only fitting for preprocessing and model selection;
external dataset evaluation;
saved checkpoints and machine-readable metrics;
automated tests;
explicit shortcut controls.
Scope and limitations

This project evaluates representation quality on public tissue-class
patches. It does not make clinical predictions and does not use patient
outcomes, treatment response, proprietary datasets or confidential
patient information.

Results are based on pilot subsets and one main training seed per model.
Internal validation and bootstrap analyses are patch-level.

License

This repository contains project source code only. Public datasets remain
subject to their original licenses and usage terms.

