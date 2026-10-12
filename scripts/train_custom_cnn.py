import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "custom_cnn"
MODEL_PATH = ROOT / "models" / "custom_cnn_best.keras"
SPLIT_PATH = ROOT / "results" / "preprocessing" / "dataset_split.csv"
WEIGHTS_PATH = ROOT / "results" / "preprocessing" / "class_weights.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("CNN_BATCH_SIZE", "16"))
MAX_EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_SEED = 42
CACHE = False
AUTOTUNE = tf.data.AUTOTUNE

metadata = pd.read_csv(SPLIT_PATH)
classes = sorted(metadata["class_name"].unique())
num_classes = len(classes)

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        tf.keras.layers.RandomRotation(0.05, seed=RANDOM_SEED + 1),
        tf.keras.layers.RandomZoom(0.1, seed=RANDOM_SEED + 2),
        tf.keras.layers.RandomBrightness(
            0.08, value_range=(0.0, 1.0), seed=RANDOM_SEED + 3
        ),
    ],
    name="training_augmentation",
)


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input((*IMAGE_SIZE, 3)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="custom_cnn",
    )
    return model


def load_image(filepath, label):
    image = tf.image.decode_image(
        tf.io.read_file(filepath), channels=3, expand_animations=False
    )
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    image.set_shape((*IMAGE_SIZE, 3))
    return image, label


def make_dataset(frame, training=False, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["filepath"].values, frame["class_index"].values)
    ).map(load_image, num_parallel_calls=AUTOTUNE)
    if CACHE:
        dataset = dataset.cache()
    if training:
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=AUTOTUNE,
        )
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def get_train_split():
    return metadata.loc[metadata["split"] == "train"]


def class_weight_mapping():
    weight_frame = pd.read_csv(WEIGHTS_PATH).sort_values("class_index")
    if weight_frame["class_index"].tolist() != list(range(num_classes)):
        raise ValueError("Training class weights do not match the class indices.")
    return dict(zip(weight_frame["class_index"], weight_frame["weight"]))


def print_stats(name, tensor):
    values = tf.cast(tensor, tf.float32)
    print(
        f"{name}: min={float(tf.reduce_min(values)):.6f}, "
        f"max={float(tf.reduce_max(values)):.6f}, "
        f"mean={float(tf.reduce_mean(values)):.6f}, "
        f"std={float(tf.math.reduce_std(values)):.6f}",
        flush=True,
    )


def check_augmentation():
    train = get_train_split()
    sample, _ = next(iter(make_dataset(train, training=False, batch_size=1)))
    image = sample[0]
    operations = [
        ("Horizontal flip", tf.keras.layers.RandomFlip("horizontal", seed=100)),
        ("Rotation", tf.keras.layers.RandomRotation(0.05, seed=101)),
        ("Zoom", tf.keras.layers.RandomZoom(0.1, seed=102)),
        (
            "Brightness, corrected normalized range",
            tf.keras.layers.RandomBrightness(
                0.08, value_range=(0.0, 1.0), seed=103
            ),
        ),
    ]
    print("Individual augmentation statistics on one real normalized train image:")
    print_stats("Input", image)
    for name, operation in operations:
        print_stats(name, operation(image, training=True))

    augmented_images, augmented_labels = next(
        iter(make_dataset(train, training=True))
    )
    print(f"Augmented batch shape: {augmented_images.shape}", flush=True)
    print(f"Augmented batch dtype: {augmented_images.dtype}", flush=True)
    print_stats("Augmented batch", augmented_images)
    batch_min = float(tf.reduce_min(augmented_images))
    batch_max = float(tf.reduce_max(augmented_images))
    if batch_min < -1e-6 or batch_max > 1.0 + 1e-6:
        raise ValueError(
            f"Augmented batch outside normalized [0, 1] range: {batch_min}, {batch_max}"
        )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    result = model.train_on_batch(augmented_images, augmented_labels)
    print(f"AUGMENTED REAL-BATCH TRAINING PASSED (diagnostic only): {result}", flush=True)


def benchmark():
    train = get_train_split()
    dataset = make_dataset(train, training=True, batch_size=BATCH_SIZE)
    iterator = iter(dataset)
    start = time.perf_counter()
    for batch_number in range(1, 6):
        images, labels = next(iterator)
        print(
            f"Benchmark batch {batch_number}/5: shape={images.shape}, "
            f"labels={labels.shape}",
            flush=True,
        )
    total_seconds = time.perf_counter() - start
    average_seconds = total_seconds / 5
    epoch_seconds = average_seconds * (len(train) / BATCH_SIZE)
    print(f"Benchmark cache enabled: {CACHE}", flush=True)
    print(f"Benchmark batches: 5", flush=True)
    print(f"Benchmark total time: {total_seconds:.3f} seconds", flush=True)
    print(f"Average seconds per batch: {average_seconds:.3f}", flush=True)
    print(f"Estimated seconds per epoch: {epoch_seconds:.1f}", flush=True)
    print(f"Estimated minutes per epoch: {epoch_seconds / 60:.2f}", flush=True)


class EpochTelemetry(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_seconds = []
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.perf_counter() - self.epoch_start
        learning_rate = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        self.epoch_seconds.append(duration)
        self.learning_rates.append(learning_rate)
        if logs is not None:
            logs["learning_rate"] = learning_rate
        print(
            f"Epoch {epoch + 1} duration: {duration:.2f}s; "
            f"learning_rate={learning_rate:.8g}",
            flush=True,
        )


def model_parameter_counts(model):
    trainable = int(sum(np.prod(variable.shape) for variable in model.trainable_weights))
    non_trainable = int(
        sum(np.prod(variable.shape) for variable in model.non_trainable_weights)
    )
    return model.count_params(), trainable, non_trainable


def curve_analysis(history):
    train_loss = history["loss"]
    val_loss = history["val_loss"]
    train_accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    best_epoch = int(np.argmin(val_loss)) + 1
    loss_gap = float(val_loss[-1] - train_loss[-1])
    accuracy_gap = float(train_accuracy[-1] - val_accuracy[-1])
    improvement = float(val_loss[0] - min(val_loss))
    convergence = (
        f"Validation loss reached its observed minimum at epoch {best_epoch} "
        f"({min(val_loss):.4f}); change from first-epoch validation loss was "
        f"{improvement:.4f}."
    )
    overfitting = (
        f"At the final epoch, validation loss minus training loss was {loss_gap:.4f}; "
        f"training accuracy minus validation accuracy was {accuracy_gap:.4f}. "
        "These endpoint gaps are descriptive, not a standalone diagnosis."
    )
    underfitting = (
        f"Final training loss was {train_loss[-1]:.4f} and training accuracy was "
        f"{train_accuracy[-1]:.4f}; interpret these together with the recorded curves."
    )
    validation = (
        f"Validation loss ranged from {min(val_loss):.4f} to {max(val_loss):.4f}; "
        f"validation accuracy ranged from {min(val_accuracy):.4f} to "
        f"{max(val_accuracy):.4f}."
    )
    return convergence, overfitting, underfitting, validation


def write_final_results(config, metrics, parameter_counts, elapsed, analysis):
    total, trainable, non_trainable = parameter_counts
    convergence, overfitting, underfitting, validation = analysis
    report = f"""# Custom CNN Final Results

## 1. Dataset

Total images: {len(metadata)}  
Classes: {num_classes}  
Training: {config['split_counts']['train']}  
Validation: {config['split_counts']['validation']}  
Testing: {config['split_counts']['test']}

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

Input(224, 224, 3) -> Conv2D(32, 3x3, same, ReLU) -> BatchNormalization -> MaxPooling2D -> Conv2D(64, 3x3, same, ReLU) -> BatchNormalization -> MaxPooling2D -> Conv2D(128, 3x3, same, ReLU) -> BatchNormalization -> MaxPooling2D -> Conv2D(256, 3x3, same, ReLU) -> BatchNormalization -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.5) -> Dense({num_classes}, softmax)

## 6. Model Complexity

Total parameters: {total}  
Trainable parameters: {trainable}  
Non-trainable parameters: {non_trainable}

## 7. Training Configuration

Optimizer: Adam  
Learning rate: {LEARNING_RATE} initial  
Batch size: {BATCH_SIZE}  
Maximum epochs: {MAX_EPOCHS}  
Actual epochs: {config['actual_epochs']}  
Loss: sparse categorical crossentropy  
Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  
Training time: {elapsed:.2f} seconds

## 8. Final Test Performance

Test Accuracy: {metrics['Test Accuracy']:.6f}  
Macro Precision: {metrics['Macro Precision']:.6f}  
Macro Recall: {metrics['Macro Recall']:.6f}  
Macro F1: {metrics['Macro F1']:.6f}  
Weighted Precision: {metrics['Weighted Precision']:.6f}  
Weighted Recall: {metrics['Weighted Recall']:.6f}  
Weighted F1: {metrics['Weighted F1']:.6f}

## 9. Training Analysis

Convergence: {convergence}  
Overfitting: {overfitting}  
Underfitting: {underfitting}  
Validation behavior: {validation}

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
"""
    (OUT / "FINAL_RESULTS.md").write_text(report, encoding="utf-8")


def train_and_evaluate():
    if BATCH_SIZE not in (16, 32):
        raise ValueError("Batch size must be 16 or 32.")
    split_counts = metadata["split"].value_counts().to_dict()
    expected = {"train": 2920, "validation": 626, "test": 626}
    if split_counts != expected:
        raise ValueError(f"Unexpected split counts: {split_counts}")
    if num_classes != 15:
        raise ValueError(f"Expected 15 classes, found {num_classes}.")

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    train_frame = get_train_split()
    validation_frame = metadata.loc[metadata["split"] == "validation"]
    train_dataset = make_dataset(train_frame, training=True)
    validation_dataset = make_dataset(validation_frame, training=False)
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Model parameters: {model.count_params()}", flush=True)
    if model.count_params() != 425167:
        raise ValueError(f"Unexpected model parameter count: {model.count_params()}")
    model.summary()

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    class_weights = class_weight_mapping()
    telemetry = EpochTelemetry()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2
        ),
        telemetry,
    ]

    training_start = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    training_seconds = time.perf_counter() - training_start
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame["epoch_duration_seconds"] = telemetry.epoch_seconds
    history_frame["learning_rate"] = telemetry.learning_rates
    history_frame.to_csv(OUT / "training_history.csv", index=False)

    for key, label, filename in [
        ("accuracy", "Accuracy", "accuracy_curve.png"),
        ("loss", "Loss", "loss_curve.png"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[key], label="Training")
        plt.plot(history.history[f"val_{key}"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / filename, dpi=160)
        plt.close()

    best_model = tf.keras.models.load_model(MODEL_PATH)
    parameter_counts = model_parameter_counts(best_model)
    pd.DataFrame(
        [
            {
                "total_parameters": parameter_counts[0],
                "trainable_parameters": parameter_counts[1],
                "non_trainable_parameters": parameter_counts[2],
                "training_time_seconds": training_seconds,
                "epochs_completed": len(history.history["loss"]),
            }
        ]
    ).to_csv(OUT / "model_complexity.csv", index=False)

    test_frame = metadata.loc[metadata["split"] == "test"]
    test_dataset = make_dataset(test_frame, training=False)
    probabilities = best_model.predict(test_dataset, verbose=1)
    predicted_labels = probabilities.argmax(axis=1)
    true_labels = test_frame["class_index"].to_numpy()
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(num_classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    class_report = pd.DataFrame(
        [
            {
                "Class": name,
                "Precision": report[name]["precision"],
                "Recall": report[name]["recall"],
                "F1-score": report[name]["f1-score"],
                "Support": int(report[name]["support"]),
            }
            for name in classes
        ]
    )
    class_report.to_csv(OUT / "classification_report.csv", index=False)

    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=list(range(num_classes))
    )
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix,
        cmap="viridis",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close()

    metrics = {
        "Model": "Custom CNN",
        "Test Accuracy": report["accuracy"],
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"],
    }
    pd.DataFrame([metrics]).to_csv(OUT / "final_metrics.csv", index=False)

    analysis = curve_analysis(history.history)
    configuration = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "keras_version": tf.keras.__version__,
        "dataset_path": str(ROOT / "PlantVillage"),
        "total_images": len(metadata),
        "number_of_classes": num_classes,
        "split_counts": {
            "train": int(split_counts["train"]),
            "validation": int(split_counts["validation"]),
            "test": int(split_counts["test"]),
        },
        "image_size": list(IMAGE_SIZE),
        "color": "RGB",
        "normalization": "pixel / 255",
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "actual_epochs": len(history.history["loss"]),
        "optimizer": "Adam",
        "initial_learning_rate": LEARNING_RATE,
        "loss": "sparse_categorical_crossentropy",
        "random_seed": RANDOM_SEED,
        "augmentation": {
            "horizontal_flip": True,
            "rotation_factor": 0.05,
            "zoom_factor": 0.1,
            "brightness_factor": 0.08,
            "brightness_value_range": [0.0, 1.0],
            "training_only": True,
        },
        "class_weights": {
            "source": str(WEIGHTS_PATH.relative_to(ROOT)),
            "derived_from_training_split_only": True,
            "used": True,
        },
        "tensorflow_threads": {
            "TF_NUM_INTRAOP_THREADS": os.environ["TF_NUM_INTRAOP_THREADS"],
            "TF_NUM_INTEROP_THREADS": os.environ["TF_NUM_INTEROP_THREADS"],
        },
        "one_dnn": os.environ["TF_ENABLE_ONEDNN_OPTS"],
        "tf_data": {
            "streaming": True,
            "map_num_parallel_calls": "AUTOTUNE",
            "prefetch": "AUTOTUNE",
            "cache": False,
        },
        "model_architecture": [
            "Input(224,224,3)",
            "Conv2D(32,3,padding=same,activation=relu)",
            "BatchNormalization",
            "MaxPooling2D",
            "Conv2D(64,3,padding=same,activation=relu)",
            "BatchNormalization",
            "MaxPooling2D",
            "Conv2D(128,3,padding=same,activation=relu)",
            "BatchNormalization",
            "MaxPooling2D",
            "Conv2D(256,3,padding=same,activation=relu)",
            "BatchNormalization",
            "GlobalAveragePooling2D",
            "Dense(128,activation=relu)",
            "Dropout(0.5)",
            f"Dense({num_classes},activation=softmax)",
        ],
        "total_parameters": parameter_counts[0],
        "trainable_parameters": parameter_counts[1],
        "non_trainable_parameters": parameter_counts[2],
        "training_time_seconds": training_seconds,
        "epoch_durations_seconds": telemetry.epoch_seconds,
        "training_analysis": {
            "convergence": analysis[0],
            "overfitting": analysis[1],
            "underfitting": analysis[2],
            "validation_behavior": analysis[3],
        },
        "test_evaluation_runs": 1,
    }
    with (OUT / "experiment_config.json").open("w", encoding="utf-8") as output:
        json.dump(configuration, output, indent=2)
    write_final_results(configuration, metrics, parameter_counts, training_seconds, analysis)
    print(f"Training seconds: {training_seconds:.2f}", flush=True)
    print(f"Epochs completed: {len(history.history['loss'])}", flush=True)
    print(f"Final test metrics: {metrics}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--check-augmentation" in sys.argv:
        check_augmentation()
    elif "--benchmark" in sys.argv:
        benchmark()
    else:
        train_and_evaluate()