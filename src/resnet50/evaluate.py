"""Evaluate once on the held-out test split."""
import argparse, json, yaml, tensorflow as tf
from .dataset import load_records, make_datasets, set_seed
def main(config, checkpoint):
    c=yaml.safe_load(open(config)); set_seed(c['seed']); r=load_records(c['data_dir'],c.get('manifest'),c.get('classes')); _,_,test,labels=make_datasets(r,c['seed'],tuple(c['image_size']),c['batch_size'],c['validation_split'],c['test_split']); print(json.dumps({'checkpoint':checkpoint,'classes':labels,'test_metrics':tf.keras.models.load_model(checkpoint).evaluate(test,return_dict=True)},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/resnet50_config.yaml'); p.add_argument('--checkpoint',required=True); a=p.parse_args(); main(a.config,a.checkpoint)
