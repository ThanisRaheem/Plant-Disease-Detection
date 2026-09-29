"""ResNet50 experiment for PlantVillage.

The model is imported lazily so lightweight EDA commands do not require the
TensorFlow runtime to be installed.
"""

__all__ = ["build_resnet50"]


def __getattr__(name):
    if name == "build_resnet50":
        from .model import build_resnet50
        return build_resnet50
    raise AttributeError(name)
