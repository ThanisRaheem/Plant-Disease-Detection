"""ResNet50 transfer-learning model."""
import tensorflow as tf
def build_resnet50(num_classes, image_size=(224,224), weights="imagenet", freeze_backbone=True):
    inputs = tf.keras.Input((*image_size,3), name="image")
    backbone = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_tensor=tf.keras.applications.resnet50.preprocess_input(inputs))
    backbone.trainable = not freeze_backbone
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(.3)(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(num_classes, activation="softmax")(x), name="resnet50_plant_disease")
