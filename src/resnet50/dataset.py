"""Deterministic loading; a shared CSV manifest can be supplied later."""
from pathlib import Path
import csv, random
import numpy as np
import tensorflow as tf
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
def load_records(data_dir, manifest=None, classes=None):
    if manifest:
        with open(manifest, newline="") as f:
            reader = csv.DictReader(f)
            required = {"path", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Manifest must contain path and label columns")
            records = [(r["path"], r["label"], r.get("split", "").lower()) for r in reader]
    else:
        root = Path(data_dir).expanduser().resolve()
        if not root.is_dir(): raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        records = [(str(p), d.name) for d in sorted(x for x in root.iterdir() if x.is_dir()) for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if classes: records = [r for r in records if r[1] in set(classes)]
    if not records: raise ValueError("No images found; check data_dir, manifest, and classes")
    return records
def make_datasets(records, seed=4050, image_size=(224,224), batch_size=32, validation_split=.15, test_split=.15):
    labels = sorted({x[1] for x in records}); mapping = {x:i for i,x in enumerate(labels)}
    has_manifest_split = all(len(record) >= 3 and record[2] in {"train", "validation", "test"} for record in records)
    if has_manifest_split:
        groups = {name: [i for i, record in enumerate(records) if record[2] == name] for name in ("train", "validation", "test")}
        def manifest_ds(name):
            return _make_dataset(records, groups[name], mapping, image_size, batch_size)
        return manifest_ds("train"), manifest_ds("validation"), manifest_ds("test"), labels
    indices = np.arange(len(records)); np.random.default_rng(seed).shuffle(indices)
    nt, nv = int(len(indices)*test_split), int(len(indices)*validation_split)
    def ds(index): return _make_dataset(records, index, mapping, image_size, batch_size)
    return ds(indices[nt+nv:]), ds(indices[nt:nt+nv]), ds(indices[:nt]), labels

def _make_dataset(records, index, mapping, image_size, batch_size):
    paths = [records[i][0] for i in index]; y = [mapping[records[i][1]] for i in index]
    d = tf.data.Dataset.from_tensor_slices((paths,y))
    def read(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(tf.cast(image, tf.float32), image_size), label
    return d.map(read, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
