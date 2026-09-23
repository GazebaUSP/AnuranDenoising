# AnuranDenoising

**A training-free filter for denoising anuran (frog/toad) recordings**, developed to assist bioacoustic species monitoring by improving classifier accuracy on cleaned recordings.

Paper: *"A Training-Free Filter for Denoising Anuran Recordings"* — submitted to ICASSP 2027.
Authors: André Gazebayukian Abrahamian, Eduardo do Valle Simões (University of São Paulo, São Carlos, SP, Brazil).

## About

Public datasets of anuran recordings are typically small, noisy, and cover many species with few samples per class — a major obstacle for training accurate species classifiers. AnuranDenoising addresses this with a **deterministic, training-free** pipeline (16 cascading signal-processing steps) that performs both:

- **Temporal segmentation** — separating croaking frames from noise, using spectral concentration and magnitude criteria around each recording's dominant frequency band.
- **Spectral cleaning** — removing environmental, anthropogenic, biological (e.g. overlapping vocalizations from other species), and recording-artifact noise via STFT-based filtering (Noisereduce with explicit noise reference + median/MAD gating).

Compared to Biodenoising, Noisereduce, MMSE-STSA and EZ-IMF, AnuranDenoising achieved the largest increase in classifier accuracy (26% → 85%), the best removal of overlapping vocalizations, and the best temporal segmentation precision, while remaining training-free.

## Repository Structure

| File | Description |
|---|---|
| `AnuranDenoising.py` | Core denoising pipeline. Takes the original raw recordings and outputs the cleaned versions (bandpass filtering, temporal segmentation, spectral cleaning, and silence trimming). |
| `cnn_optimization.py` | Hyperparameter/architecture optimization for the CNN species classifier, using Optuna to maximize accuracy via Leave-One-Out Cross-Validation (LOOCV). |
| `cnn.py` | Trains and evaluates the optimized CNN across 15 random seeds, to account for variance from model initialization. |
| `panns_optimization.py` | Hyperparameter optimization (Optuna + LOOCV) for a transfer-learning classifier based on PANNs (a large-scale pretrained audio model). |
| `panns.py` | Trains and evaluates the optimized PANNs model across 15 random seeds. |
| `requirements.txt` | Python dependencies needed to run the project. |

## Pipeline / Workflow

```
1. AnuranDenoising.py
   └── Cleans the raw recordings (denoising + trimming)

2. cnn_optimization.py / panns_optimization.py
   └── Search for the best hyperparameters (Optuna, maximizing LOOCV accuracy)

3. cnn.py / panns.py
   └── Run the optimized models across 15 seeds for robust accuracy estimates
```

## Installation

```bash
git clone https://github.com/GazebaUSP/AnuranDenoising.git
cd AnuranDenoising
pip install -r requirements.txt
```

## Usage

```bash
# 1. Denoise raw recordings
python AnuranDenoising.py

# 2. Optimize hyperparameters
python cnn_optimization.py
python panns_optimization.py

# 3. Train and evaluate final models (15 seeds each)
python cnn.py
python panns.py
```

> Adjust input/output paths inside each script as needed for your dataset.

## Results Summary

- **Classifier accuracy**: raised from 26% to 85% when training on AnuranDenoising-cleaned recordings instead of the originals — the largest gain among all compared methods.
- **CNN vs. PANNs**: noise removal eliminated the advantage of the PANNs transfer-learning model (pretrained on 5,000+ hours of audio) over a CNN trained on just 7.28 minutes of cleaned audio — both reached 85% accuracy.
- **Overlapping vocalizations**: in a case study, AnuranDenoising was the only method able to remove an overlapping vocalization from another species while preserving the masked target harmonic.
- **Temporal segmentation** (evaluated on Anuraset): eliminated 40–100% of noise while preserving 60–75% of legitimate croaking; achieved higher croaking precision than EZ-IMF (up to 100% on medium/high-quality recordings).

## Acknowledgment

This work was supported by the PUB-USP (Programa Unificado de Bolsas) scholarship. Generative AI (Claude, Anthropic) assisted in code development, literature review, and text revision; all AI-assisted content was reviewed and verified by the authors, who take full responsibility for the manuscript.

## Ethics

This study involved no direct handling of live animals; all recordings were publicly available or from third-party datasets. No ethical approval was required.