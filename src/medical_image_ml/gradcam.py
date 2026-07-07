from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer as KerasInputLayer

from medical_image_ml.paths import ensure_dir


def find_last_conv_layer(model: keras.Model) -> str:
    """Return the name of the last Conv2D / SeparableConv2D layer in the model."""
    conv_types = (layers.Conv2D, layers.SeparableConv2D)
    last_name: str | None = None
    for layer in model.layers:
        if isinstance(layer, conv_types):
            last_name = layer.name
        elif isinstance(layer, keras.Model):
            try:
                last_name = find_last_conv_layer(layer)
            except ValueError:
                continue
    if last_name is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")
    return last_name


def _make_backbone_gradcam_model(model: keras.Model, layer_name: str) -> keras.Model:
    """Build Grad-CAM model from an EfficientNet backbone submodel."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            backbone = layer
            break
    if backbone is None:
        raise ValueError("No EfficientNet backbone found for transfer Grad-CAM.")

    if layer_name is None:
        layer_name = find_last_conv_layer(backbone)

    return keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer_name).output, backbone.output],
    )


def _make_gradcam_model(model: keras.Model, layer_name: str) -> keras.Model:
    if any(isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower() for layer in model.layers):
        return _make_backbone_gradcam_model(model, layer_name)

    if not model.built:
        shape = model.input_shape
        if shape and shape[0] is None:
            dummy = np.zeros((1, *shape[1:]), dtype=np.float32)
            model(dummy, training=False)

    conv_holder: dict[str, tf.Tensor] = {}

    def walk(x: tf.Tensor, layer_list: list) -> tf.Tensor:
        for layer in layer_list:
            if isinstance(layer, KerasInputLayer):
                continue
            if isinstance(layer, keras.Model):
                x = walk(x, layer.layers)
            else:
                x = layer(x)
                if layer.name == layer_name:
                    conv_holder["conv"] = x
        return x

    inp = model.inputs[0]
    outputs = walk(inp, model.layers)
    if "conv" not in conv_holder:
        raise ValueError(f"Layer {layer_name} not reachable from model inputs.")
    return keras.Model(inputs=model.inputs, outputs=[conv_holder["conv"], outputs])


def compute_gradcam_heatmap(
    model: keras.Model,
    image: np.ndarray,
    layer_name: str | None = None,
    class_idx: int | None = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image (H, W, C) in [0, 1]."""
    use_backbone = any(
        isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower()
        for layer in model.layers
    )

    if use_backbone:
        grad_model = _make_backbone_gradcam_model(model, layer_name)
        img_batch = np.expand_dims(
            tf.image.resize(image, grad_model.input_shape[1:3]).numpy(),
            axis=0,
        )
        with tf.GradientTape() as tape:
            conv_outputs, _ = grad_model(img_batch, training=False)
            full_pred = model(np.expand_dims(image, 0), training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(full_pred[0]))
            loss = full_pred[:, class_idx]
    else:
        if layer_name is None:
            layer_name = find_last_conv_layer(model)
        grad_model = _make_gradcam_model(model, layer_name)
        img_batch = np.expand_dims(image, axis=0)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay heatmap on RGB image; returns uint8 array."""
    heatmap_resized = tf.image.resize(
        np.expand_dims(heatmap, -1),
        (image.shape[0], image.shape[1]),
        method="bilinear",
    ).numpy().squeeze()

    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap_resized)[:, :, :3]
    overlay = alpha * heatmap_color + (1 - alpha) * image
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def plot_gradcam_examples(
    model: keras.Model,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_examples: int = 4,
    output_path: str | Path | None = None,
    title: str = "Grad-CAM",
    layer_name: str | None = None,
) -> Path | None:
    """Plot original image + Grad-CAM overlay for misclassified and correct samples."""
    wrong_idx = np.where(y_true != y_pred)[0]
    correct_idx = np.where(y_true == y_pred)[0]

    chosen: list[tuple[int, str]] = []
    for idx in wrong_idx[: max(1, n_examples // 2)]:
        chosen.append((int(idx), "misclassified"))
    for idx in correct_idx[: max(1, n_examples - len(chosen))]:
        chosen.append((int(idx), "correct"))
    chosen = chosen[:n_examples]

    if not chosen:
        return None

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 2.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (idx, status) in enumerate(chosen):
        img = X[idx]
        true_l, pred_l = int(y_true[idx]), int(y_pred[idx])
        heatmap = compute_gradcam_heatmap(model, img, layer_name=layer_name, class_idx=pred_l)
        overlay = overlay_heatmap(img, heatmap)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{status}: true={true_l}, pred={pred_l}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title("Grad-CAM overlay")
        axes[row, 1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    if output_path is None:
        plt.close(fig)
        return None

    target = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target
