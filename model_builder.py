"""
Model definition.

The network is deliberately simple and split into two named parts:

    input (224x224x3, 0-255 RGB)
      -> "augmentation"  (active only while training)
      -> "backbone"      (ImageNet-pretrained CNN, include_top=False)
      -> "head"          (GAP -> Dropout -> Dense softmax over [fake, real])

Naming the backbone and head lets explainability/gradcam.py pull them back out
of a loaded .keras file without knowing which CNN was used, and keeps the
Grad-CAM code free of the nested-graph problems Keras 3 has when you try to
reach into a sub-model's layers from the outside.
"""

from __future__ import annotations

import keras
from keras import layers

import config

# keras.applications constructors and the input range each one expects.
# EfficientNet already contains Rescaling + Normalization layers, so we feed it
# raw 0-255. The others expect [-1, 1], which we add as a Rescaling layer so the
# saved model always consumes 0-255 input regardless of architecture.
_BACKBONES = {
    "efficientnetb0": (keras.applications.EfficientNetB0, None),
    "efficientnetb2": (keras.applications.EfficientNetB2, None),
    "mobilenetv2": (keras.applications.MobileNetV2, (1.0 / 127.5, -1.0)),
    "xception": (keras.applications.Xception, (1.0 / 127.5, -1.0)),
}


def available_architectures() -> list[str]:
    return sorted(_BACKBONES)


def build_backbone(architecture: str = config.ARCHITECTURE, pretrained: bool = config.PRETRAINED,
                   img_size: int = config.IMG_SIZE) -> keras.Model:
    if architecture not in _BACKBONES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {available_architectures()}")
    ctor, _ = _BACKBONES[architecture]
    weights = "imagenet" if pretrained else None
    # name="backbone" is the stable handle Grad-CAM uses to find this sub-model.
    backbone = ctor(include_top=False, weights=weights,
                    input_shape=(img_size, img_size, 3), name="backbone")
    return backbone


def build_augmentation() -> keras.Sequential:
    """Light geometric/photometric augmentation. JPEG recompression is done in the tf.data pipeline."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.08),
            layers.RandomTranslation(0.04, 0.04),
            layers.RandomBrightness(0.15, value_range=(0, 255)),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model(architecture: str = config.ARCHITECTURE, pretrained: bool = config.PRETRAINED,
                num_classes: int = 2, dropout: float = config.DROPOUT,
                img_size: int = config.IMG_SIZE, augment: bool = True) -> keras.Model:
    """Return an uncompiled functional model. Output is a softmax over class_names."""
    _, rescale = _BACKBONES[architecture]
    backbone = build_backbone(architecture, pretrained, img_size)
    if not pretrained:
        # keras.applications BatchNorm layers use momentum 0.99-0.999, tuned for
        # ImageNet-length schedules. Training from scratch on a small dataset
        # takes far fewer steps, so the moving statistics never catch up and
        # inference-mode outputs collapse (train acc 100%, val AUC 0.5). A
        # faster-moving average fixes that.
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.momentum = 0.9

    head = keras.Sequential(
        [
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax", name="classifier"),
        ],
        name="head",
    )

    inputs = keras.Input(shape=(img_size, img_size, 3), name="face_input")
    x = inputs
    if augment:
        x = build_augmentation()(x)
    if rescale is not None:
        x = layers.Rescaling(rescale[0], offset=rescale[1], name="to_minus1_1")(x)
    features = backbone(x)
    outputs = head(features)
    model = keras.Model(inputs, outputs, name=f"deepfake_{architecture}")
    return model


def set_backbone_trainable(model: keras.Model, trainable: bool, top_layers: int | None = None,
                           freeze_bn: bool = True) -> int:
    """
    Freeze/unfreeze the backbone. With `top_layers=N` only the last N layers are
    unfrozen (classic two-phase fine-tuning). BatchNorm layers stay frozen when
    fine-tuning pretrained weights (`freeze_bn=True`), which is the usual advice
    for small datasets; when training from scratch pass `freeze_bn=False`.
    Returns the number of trainable backbone layers.
    """
    backbone = model.get_layer("backbone")
    backbone.trainable = trainable
    if not trainable:
        return 0
    cutoff = 0 if top_layers is None else max(0, len(backbone.layers) - top_layers)
    count = 0
    for i, layer in enumerate(backbone.layers):
        is_bn = isinstance(layer, layers.BatchNormalization)
        if i < cutoff or (is_bn and freeze_bn):
            layer.trainable = False
        else:
            layer.trainable = True
            count += 1
    return count
