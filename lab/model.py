"""Keras model construction from a TrainConfig."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras  # noqa: E402

from lab.config import LayerSpec, TrainConfig, architecture_label, output_spec  # noqa: E402

__all__ = ["LayerSpec", "TrainConfig", "architecture_label", "build_model", "keras"]


def build_model(n_features: int, config: TrainConfig) -> keras.Model:
    keras.utils.set_random_seed(config.seed)
    regularizer = keras.regularizers.L2(config.l2) if config.l2 > 0 else None
    inputs = keras.Input(shape=(n_features,), name="input")
    x = inputs
    for i, layer in enumerate(config.layers, start=1):
        x = keras.layers.Dense(layer.units, activation=layer.activation, kernel_regularizer=regularizer, name=f"hidden_{i}")(x)
        if layer.dropout > 0:
            x = keras.layers.Dropout(layer.dropout, name=f"dropout_{i}")(x)
    out_activation, loss = output_spec(config.task)
    outputs = keras.layers.Dense(1, activation=out_activation, kernel_regularizer=regularizer, name="output")(x)
    model = keras.Model(inputs, outputs, name="mlp")
    model.compile(optimizer=_optimizer(config), loss=loss, metrics=_metrics(config.task))
    return model


def _optimizer(config: TrainConfig) -> keras.optimizers.Optimizer:
    if config.optimizer == "sgd":
        return keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum)
    return keras.optimizers.Adam(learning_rate=config.learning_rate)


def _metrics(task: str) -> list:
    if task == "regression":
        return [keras.metrics.MeanAbsoluteError(name="mae")]
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.F1Score(threshold=0.5, average="micro", name="f1"),
    ]
