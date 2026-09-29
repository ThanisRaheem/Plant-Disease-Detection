"""Train without using the test split for model selection."""
import argparse, csv, json, os, time, yaml, tensorflow as tf
from .dataset import load_records, make_datasets, set_seed
from .model import build_resnet50

class TrainingLogger(tf.keras.callbacks.Callback):
    """Capture epoch metrics, effective learning rate, and duration."""
    def __init__(self, stage):
        super().__init__(); self.stage = stage; self.rows = []; self.started = None
    def on_epoch_begin(self, epoch, logs=None): self.started = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}; learning_rate = self.model.optimizer.learning_rate
        try: learning_rate = float(tf.keras.backend.get_value(learning_rate))
        except (TypeError, ValueError): learning_rate = float(learning_rate)
        self.rows.append({"stage": self.stage, "epoch": epoch + 1, "learning_rate": learning_rate, "duration_seconds": time.perf_counter() - self.started, **{k: float(v) for k, v in logs.items()}})

def print_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Selected device: GPU ({len(gpus)} device(s); CUDA or Apple Silicon/Metal backend depending on TensorFlow installation)")
    else:
        print("Selected device: CPU")
def main(path):
    with open(path) as f: c=yaml.safe_load(f)
    print_device()
    training_started = time.perf_counter()
    set_seed(c['seed']); r=load_records(c['data_dir'],c.get('manifest'),c.get('classes'))
    tr,va,_,labels=make_datasets(r,c['seed'],tuple(c['image_size']),c['batch_size'],c['validation_split'],c['test_split'])
    m=build_resnet50(len(labels),tuple(c['image_size']),c.get('weights','imagenet'),True,c.get('augmentation',True)); os.makedirs(c['output_dir'],exist_ok=True)
    checkpoint=os.path.join(c['output_dir'],'best.keras')
    def callbacks(logger):
        return [tf.keras.callbacks.ModelCheckpoint(checkpoint,monitor='val_accuracy',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=c['patience'],restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,min_lr=1e-7,verbose=1),logger]
    logs=[]
    # Stage A: train only the newly initialized classifier.
    m.compile(optimizer=tf.keras.optimizers.Adam(c['learning_rate']),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    stage_a_logger=TrainingLogger('feature_extraction'); stage_a=m.fit(tr,validation_data=va,epochs=c['epochs'],callbacks=callbacks(stage_a_logger)); logs.extend(stage_a_logger.rows)
    # Stage B: unfreeze only the later conv5 block and use a smaller learning rate.
    for layer in m.layers:
        if layer.name == 'resnet50':
            for backbone_layer in layer.layers: backbone_layer.trainable=backbone_layer.name.startswith(c.get('unfreeze_from','conv5_block1'))
    m.compile(optimizer=tf.keras.optimizers.Adam(c.get('fine_tune_learning_rate',1e-5)),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    stage_b_logger=TrainingLogger('fine_tuning'); stage_b=m.fit(tr,validation_data=va,epochs=c.get('fine_tune_epochs',5),callbacks=callbacks(stage_b_logger)); logs.extend(stage_b_logger.rows)
    history={'stage_a':stage_a.history,'stage_b':stage_b.history}
    json.dump(labels,open(os.path.join(c['output_dir'],'labels.json'),'w'),indent=2); json.dump(history,open(os.path.join(c['output_dir'],'history.json'),'w'),indent=2)
    with open(os.path.join(c['output_dir'],'training_metrics.csv'),'w',newline='') as handle:
        fields=['stage','epoch','learning_rate','duration_seconds','loss','accuracy','val_loss','val_accuracy']; writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(logs)
    total_duration = time.perf_counter() - training_started
    json.dump({'total_training_duration_seconds': total_duration, 'epochs_completed': len(logs)}, open(os.path.join(c['output_dir'],'training_summary.json'),'w'), indent=2)
    print(f"Total training duration: {total_duration:.2f} seconds")
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/resnet50_config.yaml'); main(p.parse_args().config)
