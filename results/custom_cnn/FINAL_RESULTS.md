# Custom CNN Final Results

## 1. Dataset

Total images: 4172  
Classes: 15  
Training: 2920  
Validation: 626  
Testing: 626

## 2. Preprocessing

Original image size: 256 x 256 RGB  
Input image size: 224 x 224  
Color: RGB  
Normalization: pixel / 255

## 3. Augmentation

Horizontal flip: RandomFlip(horizontal)  
Rotation: RandomRotation(0.05)  
Zoom: RandomZoom(0.1)  
Brightness: RandomBrightness(0.08, value_range=(0, 1))

## 4. Class Weights

Training-only class weights: loaded from the existing training-derived class-weight file  
Used during training: yes

## 5. Model Architecture

Input(224, 224, 3) -> Conv2D(32, 3x3, same, ReLU) -> BatchNormalization -> MaxPooling2D -> Conv2D(64, 3x3, same, ReLU) -> BatchNormalization -> MaxPooling2D -> Conv2D(128, 3x3, same, ReLU) -> BatchNormalization -> MaxPooling2D -> Conv2D(256, 3x3, same, ReLU) -> BatchNormalization -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(15, softmax)

## 6. Model Complexity

Total parameters: 425167  
Trainable parameters: 424207  
Non-trainable parameters: 960

## 7. Training Configuration

Optimizer: Adam  
Learning rate: 0.001 initial  
Batch size: 16  
Maximum epochs: 30  
Actual epochs: 21  
Loss: sparse categorical crossentropy  
Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
Training time: 46568.64 seconds

## 8. Final Test Performance

Test Accuracy: 0.134185  
Macro Precision: 0.129296  
Macro Recall: 0.115926  
Macro F1: 0.071179  
Weighted Precision: 0.137156  
Weighted Recall: 0.134185  
Weighted F1: 0.081522

## 9. Training Analysis

Convergence: Validation loss reached its observed minimum at epoch 16 (2.5719); change from first-epoch validation loss was 1.1020.  
Overfitting: At the final epoch, validation loss minus training loss was -0.0658; training accuracy minus validation accuracy was -0.0051. These endpoint gaps are descriptive, not a standalone diagnosis.  
Underfitting: Final training loss was 2.6379 and training accuracy was 0.1291; interpret these together with the recorded curves.  
Validation behavior: Validation loss ranged from 2.5719 to 5.9631; validation accuracy ranged from 0.0639 to 0.1406.

## 10. Generated Artifacts

- `models/custom_cnn_best.keras`
- `results/custom_cnn/training_history.csv`
- `results/custom_cnn/accuracy_curve.png`
- `results/custom_cnn/loss_curve.png`
- `results/custom_cnn/model_complexity.csv`
- `results/custom_cnn/classification_report.csv`
- `results/custom_cnn/confusion_matrix.png`
- `results/custom_cnn/final_metrics.csv`
- `results/custom_cnn/experiment_config.json`
- `results/custom_cnn/FINAL_RESULTS.md`
