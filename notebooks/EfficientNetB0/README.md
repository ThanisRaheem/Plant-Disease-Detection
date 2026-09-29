# Plant Disease Classification using EfficientNetB0

## SE4050 – Group Assignment

This repository contains the EfficientNetB0 implementation for multi-class plant disease classification using the PlantVillage dataset. The model is developed using transfer learning followed by fine-tuning.

## Dataset

**Source:** https://www.kaggle.com/datasets/emmarex/plantdisease

The experiment uses 15 classes.

| Split | Images |
|---|---:|
| Training | 14,446 |
| Validation | 3,096 |
| Test | 3,096 |
| **Total** | **20,638** |

Input image size: **224 × 224 pixels**

## Model

A pretrained **EfficientNetB0** backbone is used with a classification head for the 15 PlantVillage classes.

```text
Input Image (224 × 224 × 3)
        ↓
Pretrained EfficientNetB0
        ↓
Feature Extraction
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Dense Layer
        ↓
15-Class Softmax Output
```

## Training Approach

### 1. Frozen Baseline

The pretrained EfficientNetB0 backbone is initially frozen. The newly added classification head is trained.

| Metric | Frozen Baseline |
|---|---:|
| Best Validation Accuracy | **94.28%** |
| Best Validation Loss | **0.1811** |

### 2. Fine-Tuning

Selected deeper EfficientNetB0 layers are unfrozen and trained together with the classification head using a lower learning rate. This allows the pretrained features to adapt to the plant disease classification task.

## Baseline vs Fine-Tuned

| Metric | Frozen Baseline | Fine-Tuned |
|---|---:|---:|
| Best Validation Accuracy | 94.28% | **97.45%** |
| Best Validation Loss | 0.1811 | **0.0745** |

Validation accuracy gain:

**97.45% − 94.28% = 3.17 percentage points**

Validation loss reduction:

**0.1811 − 0.0745 = 0.1066**

## Final Test Results

The final fine-tuned model was evaluated on the held-out test set of 3,096 images.

| Metric | Result |
|---|---:|
| Test Accuracy | **97.48%** |
| Macro Precision | **97.78%** |
| Macro Recall | **97.11%** |
| Macro F1-Score | **97.34%** |
| Test Loss | **0.0818** |

Classification errors:

- Total test images: **3,096**
- Incorrect predictions: **78**
- Error rate: **2.52%**

## Model Complexity

| Parameter Type | Number |
|---|---:|
| Trainable Parameters | **3,073,835** |
| Non-Trainable Parameters | **994,979** |
| **Total Parameters** | **4,068,814** |

## Inference Performance

| Metric | Result |
|---|---:|
| Batch Size | 8 |
| Time per Image | **0.01225 seconds** |
| Images per Second | **81.62** |

## Training Time

| Stage | Time |
|---|---:|
| Frozen Training | **17.93 minutes** |
| Fine-Tuning | **14.88 minutes** |
| Total | **≈32.81 minutes** |

## Experimental Workflow

```text
PlantVillage Dataset
        ↓
Data Preparation
        ↓
Train / Validation / Test Split
        ↓
Resize Images to 224 × 224
        ↓
Pretrained EfficientNetB0
        ↓
Add Classification Head
        ↓
Frozen Baseline Training
        ↓
Baseline Validation Evaluation
        ↓
Unfreeze Selected Layers
        ↓
Fine-Tuning with Lower Learning Rate
        ↓
Fine-Tuned Validation Evaluation
        ↓
Final Test Evaluation
        ↓
Metrics + Confusion Matrix + Inference Analysis
```

## Repository Structure

```text
SE4050_PlantDisease_EfficientNetB0/
│
├── README.md
├── notebooks/
│   └── EfficientNetB0_PlantDisease.ipynb
├── results/
│   ├── efficientnetb0_final_results.json
│   ├── confusion_matrix/
│   ├── training_curves/
│   └── evaluation_results/
├── models/
├── figures/
   ├── baseline_vs_finetuned_confusion_matrix.png
   ├── frozen_training_accuracy.png
   ├── frozen_training_loss.png
   ├── finetuning_accuracy.png
   └── finetuning_loss.png
```

The exact contents may change depending on the final repository organization.

## Technologies

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Git / GitHub

## Evaluation

The experiment reports:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Test loss
- Confusion matrix
- Model parameter count
- Training time
- Inference time
- Images per second

Macro F1 calculates the F1-score independently for each class and then averages the class scores.

## Reproducibility

The experiment was developed using Google Colab with a GPU-enabled runtime.

The notebook includes:

1. Dataset loading
2. Dataset preparation
3. Image preprocessing
4. Model construction
5. Frozen training
6. Fine-tuning
7. Validation evaluation
8. Test evaluation
9. Confusion matrix generation
10. Performance analysis
11. Result saving

## Results Summary

| Category | Result |
|---|---:|
| Dataset | PlantVillage |
| Classes | 15 |
| Input Size | 224 × 224 |
| Training Images | 14,446 |
| Validation Images | 3,096 |
| Test Images | 3,096 |
| Frozen Validation Accuracy | 94.28% |
| Fine-Tuned Validation Accuracy | **97.45%** |
| Validation Accuracy Gain | **3.17 pp** |
| Final Test Accuracy | **97.48%** |
| Macro Precision | **97.78%** |
| Macro Recall | **97.11%** |
| Macro F1 | **97.34%** |
| Total Parameters | **4,068,814** |
| Inference Time/Image | **0.01225 sec** |
| Images/Second | **81.62** |

## Conclusion

The EfficientNetB0 experiment demonstrates a transfer-learning workflow for multi-class plant disease classification. A frozen baseline was first established, followed by fine-tuning of selected pretrained layers.

Validation accuracy increased from **94.28% to 97.45%**, while validation loss decreased from **0.1811 to 0.0745**.

The final fine-tuned model achieved **97.48% test accuracy** and **97.34% macro F1-score** on the held-out PlantVillage test set.

The experiment also includes confusion-matrix analysis, model complexity measurements, training time, and inference performance.

## Team Project

This EfficientNetB0 implementation is one component of the overall group assignment. Each group member implements and evaluates a selected deep learning architecture using the project dataset and common evaluation approach.

---

**Model:** EfficientNetB0  
**Task:** Plant Disease Classification  
**Dataset:** PlantVillage  
**Module:** SE4050  
**Project Type:** Group Assignment
