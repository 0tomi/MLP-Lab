import time

import numpy as np

from lab.charts import network_diagram, visible_indices


def _wide_network():
    rng = np.random.default_rng(0)
    sizes = [2, 512, 512, 1]
    weights = [(rng.normal(size=(a, b)), rng.normal(size=b)) for a, b in zip(sizes[:-1], sizes[1:])]
    values = [rng.normal(size=n) for n in sizes]
    return sizes, weights, values


def test_wide_diagram_is_bounded_fast_and_shows_the_ellipsis():
    sizes, weights, values = _wide_network()
    start = time.perf_counter()
    fig = network_diagram(sizes, ["relu", "relu", "sigmoid"], [0.0, 0.0], ["x1", "x2"], weights=weights,
                          node_values=values)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert len(fig.data) <= 30
    points = sum(len(trace.x) for trace in fig.data if trace.x is not None)
    assert points <= 3 * (2 * 11 + 11 * 11 + 11 * 1) + 200  # edges between visible nodes only, plus nodes/labels
    texts = [t for trace in fig.data if trace.mode == "text" and trace.text for t in trace.text]
    assert any("+501" in t for t in texts)


def test_visible_indices_keep_the_strongest_activations_when_truncating():
    values = np.zeros(512)
    values[[3, 100, 511]] = [5.0, -7.0, 2.0]

    shown = visible_indices(512, 12, values)

    assert len(shown) == 12 and shown[-1] is None  # 11 neurons + ellipsis
    assert {3, 100, 511} <= set(shown[:-1])
    assert shown[:-1] == sorted(shown[:-1])


def test_visible_indices_without_values_show_first_and_last_neurons():
    assert visible_indices(5, 12) == [0, 1, 2, 3, 4]
    shown = visible_indices(512, 12)
    assert shown[:6] == [0, 1, 2, 3, 4, 5] and shown[6] is None and shown[7:] == [507, 508, 509, 510, 511]
