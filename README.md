# 🌿 Plant Disease Detection Using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-blue" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange" />
  <img src="https://img.shields.io/badge/Model-VGG16%20%7C%20ResNet50-green" />
  <img src="https://img.shields.io/badge/Dataset-PlantVillage-success" />
</p>

---

# 📌 Project Overview

Agriculture plays a vital role in global food production. However, plant diseases significantly reduce crop productivity and quality. Traditional disease identification methods require expert knowledge and manual inspection, which can be time-consuming and expensive.

This project develops an **Artificial Intelligence-based Plant Disease Detection System** using **Deep Learning and Transfer Learning techniques** to automatically identify plant diseases from leaf images.

The system uses advanced Convolutional Neural Network (CNN) architectures to extract visual features from plant leaves and classify them into different disease categories.

---

# 🎯 Project Objectives

The main objectives of this component are:

- Develop an automated plant disease classification system using Deep Learning.
- Apply Transfer Learning techniques for efficient feature extraction.
- Train CNN models using large-scale plant leaf image datasets.
- Compare different deep learning architectures.
- Evaluate model performance using standard evaluation metrics.
- Support early detection of plant diseases.

---

# 🏗️ System Workflow

```
                 Plant Leaf Image
                        |
                        ↓
              Image Preprocessing
                        |
                        ↓
              Image Augmentation
                        |
                        ↓
          Pre-trained CNN Architecture
              (VGG16 / ResNet50)
                        |
                        ↓
             Feature Extraction
                        |
                        ↓
              Classification Layer
                        |
                        ↓
             Disease Prediction
                        |
                        ↓
          Predicted Disease Category
```

---

# 🧠 Deep Learning Methodology

## Transfer Learning

Transfer Learning is used to improve model performance by using knowledge learned from previously trained neural networks.

The pretrained models used in this project are:

- VGG16
- ResNet50


These models are pretrained on the ImageNet dataset and adapted for plant disease classification.

---

# 🔬 Implemented Models

## 1. VGG16

VGG16 is a deep convolutional neural network consisting of 16 layers.

### Advantages:

- Simple architecture
- Effective feature extraction
- Suitable for image classification tasks


---

## 2. ResNet50

ResNet50 is a residual neural network that introduces skip connections.

### Advantages:

- Solves vanishing gradient problems
- Enables deeper network training
- Provides powerful feature extraction capability


---

# 📂 Dataset Information

## PlantVillage Dataset

The project uses the publicly available PlantVillage dataset.

Dataset contains:

- Healthy plant images
- Diseased plant images
- Multiple crop categories


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

Before training, images are processed using the following steps:

## 1. Image Resizing

All images are resized into:

```
224 × 224 × 3
```

This matches the input requirement of pretrained CNN models.

---

## 2. Image Normalization

Pixel values are normalized:

```
Normalized Pixel = Pixel Value / 255
```

This improves training stability.

---

## 3. Data Augmentation

To increase dataset diversity and prevent overfitting:

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
│   └── PlantVillage Dataset
│
├── preprocessing/
│   └── image_preprocessing.py
│
├── notebooks/
│   │
│   ├── VGG16_Training.ipynb
│   └── ResNet50_Training.ipynb
│
├── models/
│   ├── vgg16_model/
│   └── resnet50_model/
│
├── results/
│   │
│   ├── accuracy_graphs/
│   ├── confusion_matrix/
│   └── evaluation_reports/
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

## Image Processing

```
OpenCV
Pillow
```

## Data Handling

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

## Step 1: Clone Repository

```bash
git clone https://github.com/ThanisRaheem/Plant-Disease-Detection.git
```

---

## Step 2: Navigate to Project Folder

```bash
cd Plant-Disease-Detection
```

---

## Step 3: Create Virtual Environment

```bash
python -m venv .venv
```

---

## Step 4: Activate Virtual Environment


### Windows

```bash
.venv\Scripts\activate
```


### Mac/Linux

```bash
source .venv/bin/activate
```

---

## Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# 🚀 Model Training

## VGG16 Training

Run:

```bash
python train_vgg16.py
```


## ResNet50 Training

Run:

```bash
python train_resnet50.py
```

---

# 📊 Model Evaluation

The trained models are evaluated using:

## Accuracy

Measures the overall prediction correctness.

---

## Precision

Measures correctly identified disease samples.

---

## Recall

Measures the ability to detect disease cases.

---

## F1 Score

Balances precision and recall.

---

## Confusion Matrix

Provides class-level prediction analysis.

---

# 📈 Model Output

The system provides:

✅ Predicted disease class  
✅ Confidence score  
✅ Training accuracy graph  
✅ Validation accuracy graph  
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

## Component Name

**Plant Disease Detection Using Deep Learning**


## Responsibilities

- Dataset preparation
- Image preprocessing
- Data augmentation
- Deep learning model development
- Transfer learning implementation
- Model evaluation
- Performance analysis


---

# 🔮 Future Enhancements

Future improvements include:

- Real-time plant disease detection using mobile applications.
- Explainable AI integration using Grad-CAM.
- Disease severity estimation.
- Farmer recommendation system.
- Cloud-based AI deployment.
- IoT-based smart agriculture integration.


---

# 📜 Research Contribution

This component contributes towards developing an intelligent agricultural support system by applying Deep Learning methods for automatic plant disease recognition.

---

# 🙏 Acknowledgement

We would like to thank our supervisors, lecturers, and team members for their continuous guidance and support throughout this research project.

---

# 📄 License

This project is developed for academic and research purposes.
