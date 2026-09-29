# Plant Disease Detection

## Overview
This project uses deep learning techniques to detect plant diseases from leaf images.

## Features
- Image-based disease classification
- Deep learning model implementation
- Automated plant disease detection

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- CNN

## Installation

```bash
pip install -r requirements.txt
```

## ResNet50 experiment

The implementation is under `src/resnet50`. Edit
`configs/resnet50_config.yaml` to point `data_dir` at the local PlantVillage
root. The dataset is not committed. A future shared CSV split can be supplied
via `manifest` (columns `path,label`) and shared class names via `classes`.

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python -m resnet50.eda --config configs/resnet50_config.yaml
PYTHONPATH=src python -m resnet50.train --config configs/resnet50_config.yaml
PYTHONPATH=src python -m resnet50.evaluate --config configs/resnet50_config.yaml --checkpoint results/resnet50/best.keras
```

The EDA command scans the class folders without modifying images, checks every
image for corruption, prints representative dimensions/channels, and saves
class counts, summaries, sample images, and a distribution plot under
`results/resnet50/eda/`. Use `--integrity-limit N` only when a faster sampled
integrity check is preferred.

Splits use fixed seed `4050`; validation drives early stopping and model
selection, while the held-out test split is reserved for final evaluation.
