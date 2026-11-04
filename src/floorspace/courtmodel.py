"""NBA court line model in feet + helpers to project/draw it.

Coordinates: x along the court (0 = left baseline on the broadcast, 94 = right
baseline), y across (0 = far sideline, 50 = near sideline by the camera).
Each line is a polyline with a paint "kind":
  white  - lines painted white on this floor (boundary, lane, FT circle, ...)
  purple - the 3-point line (dark purple on the wood at Crypto.com Arena)
"""
from __future__ import annotations

import numpy as np

from .config import (COURT_L, COURT_W, FT_CIRCLE_R, HOOP_FROM_BASELINE, LANE_L, LANE_W,
                     RESTRICTED_R, THREE_CORNER, THREE_R, CENTER_CIRCLE_R, HALF_L)


def _arc(cx, cy, r, a0, a1, step_ft=0.5):
    n = max(8, int(abs(a1 - a0) * r / step_ft))
    a = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def _seg(p, q, step_ft=0.5):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(2, int(np.linalg.norm(q - p) / step_ft) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return p + t * (q - p)


def half_lines(side: str):
    """Lines of one half. side='left' (basket at x=0) or 'right' (x=94)."""
    lines = []
    hx = HOOP_FROM_BASELINE
    y0, y1 = (COURT_W - LANE_W) / 2, (COURT_W + LANE_W) / 2
    # lane (paint) rectangle
    lines.append(("white", "lane_far", _seg((0, y0), (LANE_L, y0))))
    lines.append(("white", "lane_near", _seg((0, y1), (LANE_L, y1))))
    lines.append(("white", "ft_line", _seg((LANE_L, y0), (LANE_L, y1))))
    # free-throw circle: the half outside the lane (the half inside is dashed, skip it)
    lines.append(("white", "ft_circle", _arc(LANE_L, COURT_W / 2, FT_CIRCLE_R, -np.pi / 2, np.pi / 2)))
    # restricted-area arc
    lines.append(("white", "restricted", _arc(hx, COURT_W / 2, RESTRICTED_R, -np.pi / 2, np.pi / 2)))
    # 3-point line: corner segments + arc
    yc0, yc1 = COURT_W / 2 - THREE_CORNER, COURT_W / 2 + THREE_CORNER
    xc = hx + np.sqrt(THREE_R ** 2 - THREE_CORNER ** 2)
    lines.append(("purple", "three_corner_far", _seg((0, yc0), (xc, yc0))))
    lines.append(("purple", "three_corner_near", _seg((0, yc1), (xc, yc1))))
    amax = np.arcsin(THREE_CORNER / THREE_R)
    lines.append(("purple", "three_arc", _arc(hx, COURT_W / 2, THREE_R, -amax, amax)))
    # boundary of this half
    lines.append(("white", "baseline", _seg((0, 0), (0, COURT_W))))
    lines.append(("white", "sideline_far", _seg((0, 0), (HALF_L, 0))))
    lines.append(("white", "sideline_near", _seg((0, COURT_W), (HALF_L, COURT_W))))
    if side == "right":
        lines = [(k, n, np.c_[COURT_L - p[:, 0], p[:, 1]]) for k, n, p in lines]
    return [(k, f"{side}_{n}", p) for k, n, p in lines]


def court_lines():
    lines = half_lines("left") + half_lines("right")
    lines.append(("white", "half_court", _seg((HALF_L, 0), (HALF_L, COURT_W))))
    lines.append(("white", "center_circle", _arc(HALF_L, COURT_W / 2, CENTER_CIRCLE_R, 0, 2 * np.pi)))
    return lines


def model_points(kinds=("white", "purple"), names_exclude=()):
    pts, kind = [], []
    for k, n, p in court_lines():
        if k in kinds and not any(s in n for s in names_exclude):
            pts.append(p)
            kind += [k] * len(p)
    return np.vstack(pts), np.array(kind)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Court feet -> image pixels."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def unproject(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Image pixels -> court feet."""
    return project(np.linalg.inv(H), uv)


def visible_mask(H: np.ndarray, pts: np.ndarray, w: int, h: int, margin: int = 0) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    uv = p[:, :2] / p[:, 2:3]
    return ((p[:, 2] > 0) & (uv[:, 0] >= -margin) & (uv[:, 0] < w + margin)
            & (uv[:, 1] >= -margin) & (uv[:, 1] < h + margin))
