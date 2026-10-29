"""Look inside a trained Keras MLP: layer sizes, weights, per-neuron activations."""

from __future__ import annotations

import numpy as np

from lab.config import count_parameters
from lab.model import keras

__all__ = ["activation_model", "count_parameters", "dense_layers", "dropout_rates", "forward_activations",
           "layer_sizes", "layer_weights"]


def dense_layers(model: keras.Model) -> list:
    return [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]


def layer_sizes(model: keras.Model) -> list[int]:
    return [int(model.inputs[0].shape[-1])] + [layer.units for layer in dense_layers(model)]


def layer_weights(model: keras.Model) -> list[tuple[np.ndarray, np.ndarray]]:
    return [tuple(layer.get_weights()) for layer in dense_layers(model)]


def activation_model(model: keras.Model) -> keras.Model:
    """Sub-model returning the output of every Dense layer (dropout is inactive at inference)."""
    return keras.Model(model.inputs[0], [layer.output for layer in dense_layers(model)])


def forward_activations(model: keras.Model, X: np.ndarray) -> list[np.ndarray]:
    """[input, hidden_1, ..., output] activations for the (already scaled) inputs X."""
    X = np.asarray(X, dtype="float32")
    outputs = activation_model(model)(X, training=False)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [X] + [np.asarray(o) for o in outputs]


def dropout_rates(model: keras.Model) -> list[float]:
    """Dropout rate applied after each hidden Dense layer (0 if none)."""
    rates, layers = [], model.layers
    for i, layer in enumerate(layers):
        if isinstance(layer, keras.layers.Dense) and layer.name != "output":
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            rates.append(float(nxt.rate) if isinstance(nxt, keras.layers.Dropout) else 0.0)
    return rates
