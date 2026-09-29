"""One-pass final evaluation on the untouched test split."""
import argparse, json, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from .dataset import load_records, make_datasets, set_seed

def plot_history(history, out):
    stages = [history.get("stage_a", {}), history.get("stage_b", {})]
    values = lambda key: sum((stage.get(key, []) for stage in stages), [])
    for key, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(values(key), label="train"); ax.plot(values("val_" + key), label="validation"); ax.set(xlabel="Epoch", ylabel=title, title=f"ResNet50 training and validation {title.lower()}"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / f"{key}_curve.png", dpi=180); plt.close(fig)

def main(config_path, checkpoint):
    with open(config_path) as f: c = yaml.safe_load(f)
    set_seed(c["seed"]); records = load_records(c["data_dir"], c.get("manifest"), c.get("classes")); _, _, test, labels = make_datasets(records, c["seed"], tuple(c["image_size"]), c["batch_size"], c["validation_split"], c["test_split"])
    model = tf.keras.models.load_model(checkpoint); output = Path(c.get("output_dir", "results/resnet50")) / "evaluation"; output.mkdir(parents=True, exist_ok=True)
    true, probabilities = [], []; started = time.perf_counter()
    for images, batch_labels in test:
        probabilities.append(model(images, training=False).numpy()); true.extend(batch_labels.numpy().tolist())
    inference_seconds = time.perf_counter() - started; y_true = np.asarray(true); y_probability = np.concatenate(probabilities); y_pred = y_probability.argmax(axis=1); class_ids = np.arange(len(labels))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=class_ids, zero_division=0); cm = confusion_matrix(y_true, y_pred, labels=class_ids); weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {"test_accuracy": float(accuracy_score(y_true, y_pred)), "test_precision_weighted": float(weighted[0]), "test_recall_weighted": float(weighted[1]), "test_f1_weighted": float(weighted[2]), "test_precision_macro": float(precision.mean()), "test_recall_macro": float(recall.mean()), "test_f1_macro": float(f1.mean()), "test_images": int(len(y_true)), "inference_seconds": inference_seconds, "inference_seconds_per_image": inference_seconds / len(y_true), "throughput_images_per_second": len(y_true) / inference_seconds, "total_parameters": int(model.count_params()), "trainable_parameters": int(sum(np.prod(v.shape) for v in model.trainable_weights)), "checkpoint_size_bytes": Path(checkpoint).stat().st_size}
    try: metrics["test_roc_auc_ovr_weighted"] = float(roc_auc_score(y_true, y_probability, multi_class="ovr", average="weighted", labels=class_ids))
    except ValueError: metrics["test_roc_auc_ovr_weighted"] = None
    metrics["per_class"] = {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)}
    json.dump(metrics, open(output / "final_metrics.json", "w"), indent=2); (output / "classification_report.txt").write_text(classification_report(y_true, y_pred, labels=class_ids, target_names=labels, zero_division=0)); np.savetxt(output / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    with open(output / "final_metrics.csv", "w") as f:
        f.write("metric,value\n"); [f.write(f"{key},{value}\n") for key, value in metrics.items() if not isinstance(value, dict)]
    fig, ax = plt.subplots(figsize=(11, 9)); image = ax.imshow(cm, cmap="Blues"); fig.colorbar(image, ax=ax); ax.set(xticks=class_ids, yticks=class_ids, xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True", title="ResNet50 confusion matrix"); plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7); plt.setp(ax.get_yticklabels(), fontsize=7); fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=180); plt.close(fig)
    x = np.arange(len(labels)); width = .25; fig, ax = plt.subplots(figsize=(13, 6)); ax.bar(x-width, precision, width, label="Precision"); ax.bar(x, recall, width, label="Recall"); ax.bar(x+width, f1, width, label="F1"); ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.05), ylabel="Score", title="Per-class performance"); plt.setp(ax.get_xticklabels(), rotation=75, ha="right", fontsize=8); ax.legend(); fig.tight_layout(); fig.savefig(output / "per_class_performance.png", dpi=180); plt.close(fig)
    history_path = Path(c.get("output_dir", "results/resnet50")) / "history.json"
    if history_path.exists(): plot_history(json.load(open(history_path)), output)
    training_summary = Path(c.get("output_dir", "results/resnet50")) / "training_summary.json"
    if training_summary.exists(): metrics["training"] = json.load(open(training_summary)); json.dump(metrics, open(output / "final_metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2)); print(f"Saved final evaluation outputs to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/resnet50_config.yaml"); parser.add_argument("--checkpoint", required=True); args = parser.parse_args(); main(args.config, args.checkpoint)
