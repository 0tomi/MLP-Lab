import numpy as np
import pytest

from lab.datasets import make_dataset, split_dataset
from lab.inspection import count_parameters, forward_activations, layer_sizes
from lab.model import LayerSpec, TrainConfig, build_model


def _config(task, layers):
    return TrainConfig(task=task, layers=tuple(layers), epochs=3, seed=0)


@pytest.mark.parametrize(
    "layers",
    [
        [],
        [LayerSpec(5, "relu")],
        [LayerSpec(6, "tanh", 0.3), LayerSpec(4, "sigmoid"), LayerSpec(3, "leaky_relu", 0.5)],
    ],
)
def test_sub_model_output_matches_predict_for_classification(layers):
    ds = make_dataset("moons", n_samples=120, noise=0.2, seed=0)
    model = build_model(n_features=2, config=_config("classification", layers))

    acts = forward_activations(model, ds.X)

    assert [a.shape[1] for a in acts] == [2] + [layer.units for layer in layers] + [1]
    np.testing.assert_allclose(acts[0], ds.X)
    np.testing.assert_allclose(acts[-1], model.predict(ds.X, verbose=0), rtol=0, atol=1e-6)


def test_sub_model_output_matches_predict_for_regression():
    ds = make_dataset("sine_1d", n_samples=80, noise=0.1, seed=0)
    model = build_model(1, _config("regression", [LayerSpec(8, "elu", 0.2), LayerSpec(4, "linear")]))

    acts = forward_activations(model, ds.X)

    np.testing.assert_allclose(acts[-1], model.predict(ds.X, verbose=0), rtol=0, atol=1e-6)


def test_sub_model_output_matches_predict_with_512_wide_layers():
    ds = make_dataset("moons", n_samples=100, noise=0.2, seed=0)
    model = build_model(2, _config("classification", [LayerSpec(512, "relu", 0.3), LayerSpec(512, "tanh")]))

    acts = forward_activations(model, ds.X)

    assert [a.shape[1] for a in acts] == [2, 512, 512, 1]
    np.testing.assert_allclose(acts[-1], model.predict(ds.X, verbose=0), rtol=0, atol=1e-6)


def test_output_layer_depends_on_task():
    clf = build_model(2, _config("classification", [LayerSpec(3, "relu")]))
    reg = build_model(2, _config("regression", [LayerSpec(3, "relu")]))
    assert clf.layers[-1].activation.__name__ == "sigmoid"
    assert reg.layers[-1].activation.__name__ == "linear"


def test_param_count_formula_matches_keras_count_params():
    ds = make_dataset("breast_cancer", n_samples=0, noise=0, seed=0)
    split = split_dataset(ds, 0.15, 0.15, 0, True)
    layers = [LayerSpec(7, "relu", 0.2), LayerSpec(5, "tanh")]
    model = build_model(split.X_train.shape[1], _config("classification", layers))

    sizes = layer_sizes(model)

    assert sizes == [30, 7, 5, 1]
    assert count_parameters(sizes) == model.count_params() == 30 * 7 + 7 + 7 * 5 + 5 + 5 * 1 + 1
