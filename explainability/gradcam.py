"""
Grad-CAM (Selvaraju et al., 2017) for the deepfake classifier.

Grad-CAM answers "WHERE did the network look?" by weighting the last
convolutional feature maps with the gradient of the chosen class score and
collapsing them into a single heatmap. It does not, by itself, say WHY - that
interpretation step lives in explain.py.

Implementation note for Keras 3: the trained model is  input -> [aug] ->
[rescale] -> backbone -> head  (see model_builder.py). Rather than trying to
build a model that reaches into the backbone's internal layers from the
outside (which Keras 3 rejects), we build a small sub-model *inside* the
backbone that returns both the Grad-CAM layer's activations and the backbone's
final features, then run the head on top under a GradientTape.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras

import config


class GradCAM:
    def __init__(self, model: keras.Model, layer_name: str | None = None):
        self.model = model
        self.backbone: keras.Model = model.get_layer("backbone")
        self.head: keras.Model = model.get_layer("head")
        self.layer_name = layer_name or self._default_layer_name()

        # Layers between the model input and the backbone (rescaling, augmentation).
        # Augmentation layers are identity at inference, Rescaling is not, so we
        # replay whatever sits in front of the backbone.
        self._pre_layers = []
        for layer in model.layers:
            if layer is self.backbone:
                break
            if isinstance(layer, keras.layers.InputLayer):
                continue
            self._pre_layers.append(layer)

        self._cam_model = keras.Model(
            inputs=self.backbone.inputs,
            outputs=[self.backbone.get_layer(self.layer_name).output, self.backbone.output],
            name="gradcam_backbone",
        )

    def _default_layer_name(self) -> str:
        for arch, name in config.GRADCAM_LAYER.items():
            if arch in self.model.name:
                try:
                    self.backbone.get_layer(name)
                    return name
                except ValueError:
                    pass
        # Fallback: last 4-D output in the backbone.
        for layer in reversed(self.backbone.layers):
            shape = getattr(layer, "output", None)
            shape = getattr(shape, "shape", None)
            if shape is not None and len(shape) == 4:
                return layer.name
        raise ValueError("Could not find a convolutional layer for Grad-CAM")

    def _forward_pre(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self._pre_layers:
            x = layer(x, training=False)
        return x

    def compute(self, image: np.ndarray, class_index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        image: preprocessed (H, W, 3) float32 array (output of preprocess_face)
        class_index: which softmax output to explain
        Returns (heatmap in [0,1] at conv resolution, softmax probabilities).
        """
        x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
        x = self._forward_pre(x)

        with tf.GradientTape() as tape:
            conv_out, features = self._cam_model(x, training=False)
            tape.watch(conv_out)
            probs = self.head(features, training=False)
            score = probs[:, class_index]

        grads = tape.gradient(score, conv_out)              # (1, h, w, c)
        weights = tf.reduce_mean(grads, axis=(1, 2))        # (1, c)  - global-average-pooled gradients
        cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]  # (h, w)
        cam = tf.nn.relu(cam).numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32), probs.numpy()[0]

    def compute_batch(self, images: np.ndarray, class_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised version for several images with the same target class."""
        x = tf.convert_to_tensor(images, dtype=tf.float32)
        x = self._forward_pre(x)
        with tf.GradientTape() as tape:
            conv_out, features = self._cam_model(x, training=False)
            tape.watch(conv_out)
            probs = self.head(features, training=False)
            score = probs[:, class_index]
        grads = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cams = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
        cams = tf.nn.relu(cams).numpy()
        maxes = cams.reshape(len(cams), -1).max(axis=1)
        maxes[maxes == 0] = 1.0
        cams = cams / maxes[:, None, None]
        return cams.astype(np.float32), probs.numpy()
