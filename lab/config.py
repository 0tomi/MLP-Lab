"""Pure model configuration (no TensorFlow import, so the UI paints fast)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

ACTIVATIONS = ("linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu")
OPTIMIZERS = ("adam", "sgd")

# Discrete options offered by the UI; imported values snap to the nearest one.
LEARNING_RATES = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
BATCH_SIZES = (1, 8, 16, 32, 64, 128, 256)
L2_VALUES = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
MAX_HIDDEN = 4
MAX_UNITS = 512

# Default limit: above this many parameters the decision boundary / fitted curve is drawn only when
# training ends. Conservative for average machines; override per machine in local_settings.json or the UI.
# Measured on a fast machine: one redraw costs ~8 ms at 1k params, ~18 ms at 67k, ~36 ms at 265k and
# ~57 ms at 527k (plus ~220 KB sent to the browser), against ~65-75 ms per training epoch.
LIVE_BOUNDARY_MAX_PARAMS = 15_000


def should_draw_live_boundary(n_params: int, max_params: int) -> bool:
    """0 means the boundary is always drawn only at the end."""
    return 0 < max_params and n_params <= max_params

# monitor name -> EarlyStopping mode
MONITOR_OPTIONS: dict[str, dict[str, str]] = {
    "classification": {
        "val_loss": "min",
        "val_accuracy": "max",
        "val_auc": "max",
        "val_f1": "max",
        "val_precision": "max",
        "val_recall": "max",
    },
    "regression": {"val_loss": "min", "val_mae": "min"},
}


@dataclass(frozen=True)
class LayerSpec:
    units: int
    activation: str = "relu"
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    task: Literal["classification", "regression"]
    layers: tuple[LayerSpec, ...] = field(default_factory=lambda: (LayerSpec(8, "relu"),))
    optimizer: str = "adam"
    learning_rate: float = 0.01
    momentum: float = 0.0
    batch_size: int = 32
    epochs: int = 200
    l2: float = 0.0
    monitor: str = "val_loss"
    patience: int = 30
    early_stopping: bool = True
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self) | {"layers": [asdict(layer) for layer in self.layers]}

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        return cls(**(data | {"layers": tuple(LayerSpec(**layer) for layer in data["layers"])}))


def output_spec(task: str) -> tuple[str, str]:
    """(output activation, loss) fixed by the task."""
    return ("sigmoid", "binary_crossentropy") if task == "classification" else ("linear", "mse")


def layer_sizes_of(n_features: int, layers) -> list[int]:
    return [n_features] + [_spec(layer).units for layer in layers] + [1]


def count_parameters(sizes: list[int]) -> int:
    """Σ (n_l · n_{l-1} + n_l): weights plus one bias per neuron."""
    return sum(n * prev + n for prev, n in zip(sizes[:-1], sizes[1:]))


def architecture_label(n_features: int, layers) -> str:
    """Compact summary such as 2-8(relu,d0.2)-4(tanh)-1."""
    parts = [str(n_features)]
    for layer in map(_spec, layers):
        drop = f",d{layer.dropout:g}" if layer.dropout > 0 else ""
        parts.append(f"{layer.units}({layer.activation}{drop})")
    return "-".join(parts + ["1"])


def _spec(layer) -> LayerSpec:
    return layer if isinstance(layer, LayerSpec) else LayerSpec(**layer)
