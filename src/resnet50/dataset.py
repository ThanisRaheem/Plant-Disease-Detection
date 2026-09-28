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
        with open(manifest, newline="") as f: records = [(r["path"], r["label"]) for r in csv.DictReader(f)]
    else:
        root = Path(data_dir).expanduser().resolve()
        if not root.is_dir(): raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        records = [(str(p), d.name) for d in sorted(x for x in root.iterdir() if x.is_dir()) for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if classes: records = [r for r in records if r[1] in set(classes)]
    if not records: raise ValueError("No images found; check data_dir, manifest, and classes")
    return records
def make_datasets(records, seed=4050, image_size=(224,224), batch_size=32, validation_split=.15, test_split=.15):
    labels = sorted({x[1] for x in records}); mapping = {x:i for i,x in enumerate(labels)}
    indices = np.arange(len(records)); np.random.default_rng(seed).shuffle(indices)
    nt, nv = int(len(indices)*test_split), int(len(indices)*validation_split)
    def ds(index):
        paths = [records[i][0] for i in index]; y = [mapping[records[i][1]] for i in index]
        d = tf.data.Dataset.from_tensor_slices((paths,y))
        def read(path, label):
            image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
            return tf.image.resize(tf.cast(image, tf.float32), image_size), label
        return d.map(read, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds(indices[nt+nv:]), ds(indices[nt:nt+nv]), ds(indices[:nt]), labels
