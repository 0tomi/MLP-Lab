import json

import pytest

from lab.settings import load_live_boundary_max_params, save_live_boundary_max_params


def test_missing_file_gives_the_default(tmp_path):
    assert load_live_boundary_max_params(tmp_path / "local_settings.json") == 15_000


def test_valid_file_overrides_the_default(tmp_path):
    path = tmp_path / "local_settings.json"
    path.write_text(json.dumps({"live_boundary_max_params": 50_000}))
    assert load_live_boundary_max_params(path) == 50_000


@pytest.mark.parametrize(
    "content",
    ["{not json", "[]", '{"live_boundary_max_params": -1}', '{"live_boundary_max_params": "lots"}',
     '{"live_boundary_max_params": 20000000}', '{"live_boundary_max_params": true}', "{}"],
)
def test_invalid_file_falls_back_to_the_default(tmp_path, content):
    path = tmp_path / "local_settings.json"
    path.write_text(content)
    assert load_live_boundary_max_params(path) == 15_000


def test_save_then_load_round_trip_keeps_other_keys(tmp_path):
    path = tmp_path / "local_settings.json"
    path.write_text(json.dumps({"other": 1}))
    save_live_boundary_max_params(0, path)
    assert load_live_boundary_max_params(path) == 0
    assert json.loads(path.read_text())["other"] == 1
