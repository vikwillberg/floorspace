"""Unit tests for the spacing metrics (no video, models or GPU needed)."""
import math

import numpy as np
import pandas as pd
import pytest

from floorspace import config, metrics


def test_hull_area_square():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]])   # inner point ignored
    assert metrics.hull_area(pts) == pytest.approx(100.0)


def test_hull_area_triangle_and_degenerate():
    assert metrics.hull_area(np.array([[0, 0], [4, 0], [0, 3]])) == pytest.approx(6.0)
    assert metrics.hull_area(np.array([[0, 0], [1, 1]])) == 0.0
    assert metrics.hull_area(np.array([[0, 0], [1, 1], [2, 2]])) == pytest.approx(0.0)


def test_hull_never_exceeds_half_court():
    corners = np.array([[47, 0], [94, 0], [94, 50], [47, 50], [70, 25]])
    assert metrics.hull_area(corners) == pytest.approx(config.HALF_L * config.COURT_W)
    assert metrics.hull_area(corners) <= config.HALF_COURT_AREA


def test_mean_pairwise():
    pts = np.array([[0, 0], [3, 4], [6, 8]])            # distances 5, 5, 10
    assert metrics.mean_pairwise(pts) == pytest.approx(20 / 3)


def test_outside_three_left_basket():
    hx, mid = config.HOOP_FROM_BASELINE, config.COURT_W / 2
    x = np.array([hx, hx + 23.0, hx + 24.5, 3.0, 3.0])
    y = np.array([mid, mid, mid, mid + 21.5, mid + 23.0])
    # rim, inside top of arc, outside top of arc, inside corner, outside corner
    assert metrics.outside_three(x, y, "left").tolist() == [False, False, True, False, True]


def test_outside_three_is_mirrored_for_right_basket():
    x = np.array([10.0, 30.0, 3.0])
    y = np.array([25.0, 25.0, 48.0])
    left = metrics.outside_three(x, y, "left")
    right = metrics.outside_three(config.COURT_L - x, y, "right")
    assert left.tolist() == right.tolist()


def test_per_frame_needs_all_five():
    rows = []
    for f in range(2):
        n = 5 if f == 0 else 4
        for p in range(n):
            rows.append(dict(frame=f, pid=p, team="LAL", state="observed",
                             x_ft=60 + 5 * p, y_ft=10 + 7 * p))
    sm = pd.DataFrame(rows)
    info = dict(offense="LAL", window_start_s=0, window_end_s=10)
    sp = metrics.per_frame("t", sm, info, 2)
    assert sp.loc[0, "all5_tracked"] and not sp.loc[1, "all5_tracked"]
    assert math.isnan(sp.loc[1, "hull_area_sqft"])
