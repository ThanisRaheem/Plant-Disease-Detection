# 🌿 Plant Disease Detection Using Deep Learning

<p align="center">
<img src="https://img.shields.io/badge/AI-Deep%20Learning-blue">
<img src="https://img.shields.io/badge/Framework-TensorFlow-orange">
<img src="https://img.shields.io/badge/Models-CNN%20%7C%20VGG16%20%7C%20ResNet50%20%7C%20EfficientNetB0-green">
<img src="https://img.shields.io/badge/Dataset-PlantVillage-success">
</p>


# 📌 Project Overview

Agriculture is one of the most important sectors that supports global food production. Plant diseases significantly affect crop productivity and create economic losses for farmers.

Traditional plant disease identification requires manual observation by agricultural experts, which can be time-consuming and difficult to perform on a large scale.

This project develops an **Artificial Intelligence-based Plant Disease Detection System** using **Deep Learning and Transfer Learning techniques** to automatically identify diseases from plant leaf images.

The system applies multiple Convolutional Neural Network (CNN) architectures and compares their performance for accurate plant disease classification.


The implemented models include:

- Custom CNN
- VGG16
- ResNet50
- EfficientNetB0


---

# 🎯 Project Objectives

The main objectives of this component are:

- Develop an AI-based plant disease classification system.
- Perform image preprocessing and augmentation.
- Implement multiple deep learning architectures.
- Apply transfer learning using pretrained CNN models.
- Compare model performance using evaluation metrics.
- Identify the most effective model for plant disease detection.


---

# 🏗️ System Workflow


```
                Plant Leaf Image
                       |
                       ↓
             Image Preprocessing
                       |
                       ↓
              Data Augmentation
                       |
                       ↓

        ---------------------------------
        |        |          |           |
        ↓        ↓          ↓           ↓

      CNN     VGG16    ResNet50   EfficientNetB0

        |        |          |           |
        ---------------------------------

                       |
                       ↓

             Feature Extraction

                       |
                       ↓

              Classification Layer

                       |
                       ↓

          Plant Disease Prediction

```


---

# 🧠 Deep Learning Methodology


This research investigates different CNN-based architectures for plant disease classification.

The models are divided into:


## 1. Custom Deep Learning Model

A CNN architecture is created from scratch to establish a baseline performance.


## 2. Transfer Learning Models

Pretrained models are adapted from ImageNet and fine-tuned for plant disease classification.


Implemented transfer learning models:

- VGG16
- ResNet50
- EfficientNetB0


---

# 🔬 Implemented Deep Learning Models


# 1. Custom CNN Model


## Overview

A custom Convolutional Neural Network is developed as a baseline model.

The network learns important visual patterns from plant leaf images including:

- Edges
- Shapes
- Textures
- Disease patterns


## Architecture Components

```
Input Image

↓

Convolution Layer

↓

Activation Function

↓

Pooling Layer

↓

Convolution Layer

↓

Dropout

↓

Fully Connected Layer

↓

Softmax Output

```


## Purpose

- Establish baseline classification performance.
- Understand CNN feature learning.
- Compare against advanced architectures.


---


# 2. VGG16 Transfer Learning Model


## Overview

VGG16 is a deep convolutional neural network developed by the Visual Geometry Group.


## Architecture

VGG16 contains:

- 13 Convolution Layers
- 5 Max Pooling Layers
- Fully Connected Layers


## Implementation

The pretrained ImageNet model is adapted by:

- Removing original classifier layers.
- Adding custom classification layers.
- Training on plant disease images.


## Advantages

- Simple architecture.
- Strong feature extraction ability.
- Widely used in image classification.


---


# 3. ResNet50 Transfer Learning Model


## Overview

ResNet50 is a residual neural network architecture designed to train deeper networks efficiently.


## Key Feature

Residual connections allow information to skip layers:

```
Input

↓

Convolution Layers

↓

+

↓

Output

```


## Advantages

- Reduces vanishing gradient problems.
- Learns complex image features.
- Supports deeper architectures.


## Implementation

The model uses:

- ImageNet pretrained weights.
- Custom classification layer.
- Fine-tuning strategy.


---


# 4. EfficientNetB0 Transfer Learning Model


## Overview

EfficientNetB0 is a modern CNN architecture that balances network depth, width, and image resolution using compound scaling.


## Key Features

- Efficient feature extraction.
- Fewer parameters.
- Better computational efficiency.


## Advantages

- High accuracy with lower computational cost.
- Suitable for real-world AI applications.
- Effective for image classification tasks.


## Implementation

The model uses:

- ImageNet pretrained EfficientNetB0 backbone.
- Custom output classification layer.
- Plant disease dataset training.


---

# 📂 Dataset Information


## PlantVillage Dataset


This project uses the PlantVillage dataset containing thousands of plant leaf images.


Dataset includes:

- Healthy plant images.
- Diseased plant images.
- Multiple crop categories.


Example classes:

```
Apple___Apple_scab

Apple___Black_rot

Apple___Healthy

Potato___Early_blight

Potato___Late_blight

Tomato___Bacterial_spot

Tomato___Late_blight

```


---

# 🔄 Image Preprocessing Pipeline


Before model training, images are processed using:


## 1. Image Resizing


All images are resized into:


```
224 × 224 × 3
```


to match pretrained CNN input requirements.


---

## 2. Normalization


Pixel values are scaled:


```
Normalized Pixel = Pixel Value / 255
```


---

## 3. Data Augmentation


To improve model generalization:


Applied techniques:

- Rotation
- Horizontal Flip
- Zoom
- Brightness Adjustment
- Width Shift
- Height Shift


---

# 📁 Project Structure


```
Plant-Disease-Detection/

│
├── dataset/
│
├── preprocessing/
│
├── notebooks/
│   │
│   ├── CNN_training.ipynb
│   ├── VGG16_training.ipynb
│   ├── ResNet50_training.ipynb
│   └── EfficientNetB0_training.ipynb
│
├── models/
│   │
│   ├── cnn/
│   ├── vgg16/
│   ├── resnet50/
│   └── efficientnetb0/
│
├── results/
│   │
│   ├── accuracy/
│   ├── loss/
│   ├── confusion_matrix/
│   └── reports/
│
├── requirements.txt
│
└── README.md

```


---

# 🛠️ Technologies Used


## Programming Language

```
Python 3.x
```


## Deep Learning Framework

```
TensorFlow
Keras
```


## Deep Learning Models

```
Custom CNN

VGG16

ResNet50

EfficientNetB0
```


## Image Processing

```
OpenCV

Pillow
```


## Data Processing

```
NumPy

Pandas
```


## Visualization

```
Matplotlib

Seaborn
```


---

# ⚙️ Installation Guide


## Clone Repository


```bash
git clone https://github.com/ThanisRaheem/Plant-Disease-Detection.git
```


## Navigate Project Folder


```bash
cd Plant-Disease-Detection
```


## Create Virtual Environment


```bash
python -m venv .venv
```


## Activate Environment


### Windows

```bash
.venv\Scripts\activate
```


### Mac/Linux

```bash
source .venv/bin/activate
```


## Install Dependencies


```bash
pip install -r requirements.txt
```


---

# 🚀 Model Training


## Custom CNN Training


```bash
python train_cnn.py
```


## VGG16 Training


```bash
python train_vgg16.py
```


## ResNet50 Training


```bash
python train_resnet50.py
```


## EfficientNetB0 Training


```bash
python train_efficientnetb0.py
```


---

# 📊 Model Evaluation


The trained models are evaluated using:


## Accuracy

Measures overall classification performance.


## Precision

Measures correctness of positive predictions.


## Recall

Measures disease detection capability.


## F1-score

Provides balance between precision and recall.


## Confusion Matrix

Shows class-level prediction performance.


## Training Curves

Used to analyze:

- Accuracy improvement
- Loss reduction
- Overfitting behaviour


---

# 📈 System Output


The system provides:


✅ Predicted disease category  
✅ Model confidence score  
✅ Training results  
✅ Validation performance  
✅ Confusion matrix  
✅ Classification report  


Example:


```
Input:

Tomato Leaf Image


Prediction:

Tomato___Late_blight


Confidence:

95.6%

```


---

# 👥 Team Contribution


## Component

**Plant Disease Detection Using Deep Learning**


Responsibilities:


- Dataset preparation
- Image preprocessing
- CNN implementation
- Transfer learning implementation
- Model training
- Model comparison
- Performance evaluation


---

# 🔮 Future Enhancements


Future improvements include:


- Explainable AI using Grad-CAM.
- Mobile application integration.
- Real-time camera-based disease detection.
- Disease severity prediction.
- Smart agriculture recommendation system.
- Cloud deployment.


---

# 🙏 Acknowledgement


We would like to thank our supervisors, lecturers, and team members for their valuable guidance and continuous support throughout this research project.


---

# 📜 License


This project is developed for academic and research purposes.
