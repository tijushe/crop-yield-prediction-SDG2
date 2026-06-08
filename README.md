# Crop Yield Prediction with Transfer Learning and Satellite Imagery

A deep learning project that extends [Wang et al. (2018)](https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction) by evaluating how well an LSTM model trained on Argentine soybean data can transfer to new geographies and crop types, and whether adding environmental features improves prediction accuracy.

Built as part of a university course project at the Leiden University (2024), using the [SustainBench](https://sustainlab-group.github.io/sustainbench/) benchmark framework and satellite imagery from [NASA MODIS](https://modis.gsfc.nasa.gov/).

---

## Project Overview

Accurately predicting crop yields is critical for food security policy, especially in data-scarce regions. This project builds on transfer learning — the idea that a model trained on one region or crop can be fine-tuned for another with far less data.

We trained an LSTM baseline on Argentine soybean yield data using satellite histogram features, then ran four experiments to stress-test its transferability:

| Experiment | Description |
|---|---|
| 1 | Transfer learning: Argentina soybean → **US soybean** |
| 2 | Transfer learning: Argentina soybean → **US corn** |
| 3 | Transfer learning: Argentina soybean → **Argentina corn** |
| 4 | Feature enrichment: Adding **rainfall data** to the Argentina LSTM model |

---

## Results Summary

| Task | R² | RMSE |
|---|---|---|
| Argentina Soybean (baseline) | — | 0.779 |
| US Soybean (transfer) | -0.011 | 0.687 |
| US Corn (transfer + fine-tune) | 0.267 | 2.409 |
| Argentina LSTM + Rainfall | — | — |

Transfer to the US proved challenging, reflecting the scale and agricultural diversity differences between countries. Adding rainfall as an additional feature was explored as a way to improve accuracy beyond satellite imagery alone.

---

## Repository Structure

```
├── crop-yield-prediction-sdg2-1.ipynb   # Experiments 1, 2, and 4 (US soybean, US corn, rainfall features)
├── crop-yield-prediction-sdg2-2.ipynb     # Experiment 3 (Argentina corn transfer)
├── pretrained_model.weights.h5  # Pre-trained LSTM weights (Argentina soybean)
├── rainfall.csv                 # Rainfall data used in Experiment 4
├── req_cybench.md               # Setup instructions for CY-Bench dataset
└── Report.pdf   # Final written report
```

---

## Getting Started

### Prerequisites

- Python 3.9
- TensorFlow 2.x
- Google Colab (recommended — notebooks are set up for Colab with Google Drive)

### Data

This project uses:
- **SustainBench** crop yield dataset — [setup guide here](https://sustainlab-group.github.io/sustainbench/docs/get_started.html)
- **CY-Bench** dataset for US corn/soybean data:

```bash
git clone https://github.com/BigDataWUR/AgML-CY-Bench
pip install poetry
cd AgML-CY-Bench
poetry install

# Download sample data
git clone https://github.com/BigDataWUR/sample_data.git cybench/data
```

### Running the Notebooks

Open either notebook in Google Colab. The setup cells will mount your Google Drive and install dependencies automatically. Run cells sequentially.

The pre-trained model weights (`pretrained_model.weights.h5`) can be loaded directly to skip retraining the baseline.

---

## Methods

- **Model**: LSTM neural network, adapted from Wang et al. (2018) and updated for TensorFlow 2.x
- **Input features**: 9-band satellite image histograms (32 timesteps × 32 bins × 9 spectral bands), plus rainfall in Experiment 4
- **Transfer learning**: Pre-train on Argentina soybean → freeze base layers → fine-tune on target dataset
- **Hyperparameters** (tuned via grid search): dropout 0.75, learning rate 0.01, 200 LSTM units
- **Metrics**: RMSE and R²

---

## References

- Wang et al. (2018). *Deep Transfer Learning for Crop Yield Prediction with Remote Sensing Data*. ACM COMPASS. [GitHub](https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction)
- Yeh et al. (2021). *SustainBench: Benchmarks for Monitoring the SDGs with Machine Learning*. [arXiv](https://arxiv.org/abs/2111.04724)
- Lu et al. (2024). *GOA-optimized deep learning for soybean yield estimation using multi-source remote sensing data*. Scientific Reports.

---

## Authors

**Senjuti Bala** and **Jiamian He** — Leiden University, 2024
