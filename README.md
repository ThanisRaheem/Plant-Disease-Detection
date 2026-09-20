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

Create the environment with Python 3.13 (on this Mac, use
`/opt/anaconda3/bin/python` in place of `python3.13`):

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

In VS Code, open `VGG16_Plant_Disease.ipynb`, click **Select Kernel**,
choose **Python Environments**, and select `.venv/bin/python`.
If it is not listed, run **Python: Select Interpreter** from the Command
Palette and enter that interpreter path, then select it as the notebook kernel.
Restart the notebook kernel before running the cells.

Use the project environment instead of Anaconda's `base` environment.
The original base environment crashed during `import tensorflow`; the macOS
crash report showed a native Protobuf segmentation fault involving
`libtensorflow_framework.2.dylib` and Anaconda's `libprotobuf.29.3.0.dylib`.
