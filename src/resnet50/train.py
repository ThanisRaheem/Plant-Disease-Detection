"""Train without using the test split for model selection."""
import argparse, json, os, yaml, tensorflow as tf
from .dataset import load_records, make_datasets, set_seed
from .model import build_resnet50
def main(path):
    with open(path) as f: c=yaml.safe_load(f)
    set_seed(c['seed']); r=load_records(c['data_dir'],c.get('manifest'),c.get('classes'))
    tr,va,_,labels=make_datasets(r,c['seed'],tuple(c['image_size']),c['batch_size'],c['validation_split'],c['test_split'])
    m=build_resnet50(len(labels),tuple(c['image_size']),c.get('weights','imagenet'),c.get('freeze_backbone',True)); m.compile(optimizer=tf.keras.optimizers.Adam(c['learning_rate']),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    os.makedirs(c['output_dir'],exist_ok=True); cb=[tf.keras.callbacks.ModelCheckpoint(os.path.join(c['output_dir'],'best.keras'),monitor='val_accuracy',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=c['patience'],restore_best_weights=True)]
    h=m.fit(tr,validation_data=va,epochs=c['epochs'],callbacks=cb)
    json.dump(labels,open(os.path.join(c['output_dir'],'labels.json'),'w'),indent=2); json.dump(h.history,open(os.path.join(c['output_dir'],'history.json'),'w'),indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/resnet50_config.yaml'); main(p.parse_args().config)
