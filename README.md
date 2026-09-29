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

## Shared dataset split

No permanent split is assumed until the group agrees on the shared classes and
ratios. Create one manifest once, then give the same CSV to every model:

```bash
PYTHONPATH=src python -m resnet50.split \
  --data-dir /path/to/PlantVillage \
  --output /path/to/shared_split.csv --seed 42
```

The generator assigns each class to 70% train, 15% validation, and 15% test
using a fixed seed, without copying image files. A manifest must contain
`path,label,split`; the loader uses those assignments exactly and never
reshuffles them. The test rows are only used by the evaluation command.

## ResNet50 preprocessing

Images are resized to **224×224 RGB**. The model applies Keras
`ResNet50.preprocess_input`, which converts RGB pixels to the channel convention
and ImageNet training scale expected by the pretrained backbone. During
training only, the configured augmentation pipeline applies a horizontal flip,
small rotation (±9°), modest zoom (10%), small translation (5%), and conservative
contrast variation (10%). These transformations are applied before ImageNet
normalization. Validation and test data use only resize and RGB conversion;
they contain no random augmentation, so model selection and final evaluation
remain deterministic and comparable across architectures.

Splits use fixed seed `4050`; validation drives early stopping and model
selection, while the held-out test split is reserved for final evaluation.
