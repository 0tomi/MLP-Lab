from lab.config import LIVE_BOUNDARY_MAX_PARAMS, should_draw_live_boundary


def test_default_live_boundary_threshold_is_15k():
    assert LIVE_BOUNDARY_MAX_PARAMS == 15_000


def test_live_boundary_threshold_is_a_parameter():
    assert should_draw_live_boundary(14_999, 15_000) is True
    assert should_draw_live_boundary(15_000, 15_000) is True
    assert should_draw_live_boundary(15_001, 15_000) is False
    assert should_draw_live_boundary(264_705, 50_000) is False
    assert should_draw_live_boundary(1, 0) is False  # 0: always only at the end
