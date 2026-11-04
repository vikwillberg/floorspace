"""Floor-spacing metrics (court feet, offense only).

Per frame (only when all 5 offensive players are tracked, otherwise flagged):
  hull_area_sqft   area of the convex hull of the 5 offensive players' feet
  mean_dist_ft     mean of the 10 pairwise distances between offensive teammates
  n_outside_3      offensive players behind the 3-point line (feet on the line = inside)

"Tracked" = position observed by the detector or interpolated by the smoother
across a gap of at most 0.8 s. Players only held at their last position (lost
for good) do not count.

Per clip (inside the analysis window from clips.csv): average and minimum of each
metric, and the average over the last 1 s before the shot release.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config

TRACKED_STATES = ("observed", "interpolated")


def hull_area(pts: np.ndarray) -> float:
    """Convex hull area (monotone chain + shoelace)."""
    p = sorted(map(tuple, np.asarray(pts, float)))
    if len(p) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    h = np.array(lower[:-1] + upper[:-1])
    x, y = h[:, 0], h[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def hull_polygon(pts: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:
        return pts


def mean_pairwise(pts: np.ndarray) -> float:
    pts = np.asarray(pts, float)
    d = [np.linalg.norm(pts[i] - pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))]
    return float(np.mean(d)) if d else float("nan")


def outside_three(x: np.ndarray, y: np.ndarray, side: str) -> np.ndarray:
    """True if the point is behind the 3-point line of the basket on `side`."""
    xb = np.asarray(x, float) if side == "left" else config.COURT_L - np.asarray(x, float)
    y = np.asarray(y, float)
    hx = config.HOOP_FROM_BASELINE
    corner_end = hx + math.sqrt(config.THREE_R ** 2 - config.THREE_CORNER ** 2)   # ~14.2 ft
    dy = np.abs(y - config.COURT_W / 2)
    in_corner_zone = xb <= corner_end
    out_corner = in_corner_zone & (dy > config.THREE_CORNER)
    out_arc = ~in_corner_zone & (np.hypot(xb - hx, y - config.COURT_W / 2) > config.THREE_R)
    return out_corner | out_arc


def attack_side(sm: pd.DataFrame, offense: str) -> str:
    x = sm[(sm.team == offense) & sm.state.isin(TRACKED_STATES)].x_ft
    return "right" if x.median() > config.HALF_L else "left"


def per_frame(clip: str, sm: pd.DataFrame, info: dict, n_frames: int) -> pd.DataFrame:
    off = info["offense"]
    side = attack_side(sm, off)
    w0, w1 = float(info["window_start_s"]), float(info["window_end_s"])
    rows = []
    for f in range(n_frames):
        g = sm[(sm.frame == f) & (sm.team == off) & sm.state.isin(TRACKED_STATES)]
        t = f / config.FPS
        r = dict(clip=clip, frame=f, t_sec=round(t, 3), offense=off, attack_side=side,
                 in_window=bool(w0 <= t <= w1), n_offense_tracked=len(g))
        if len(g) == 5:
            pts = g[["x_ft", "y_ft"]].to_numpy()
            r.update(all5_tracked=True, hull_area_sqft=round(hull_area(pts), 1),
                     mean_dist_ft=round(mean_pairwise(pts), 2),
                     n_outside_3=int(outside_three(pts[:, 0], pts[:, 1], side).sum()))
        else:
            r.update(all5_tracked=False, hull_area_sqft=np.nan, mean_dist_ft=np.nan, n_outside_3=np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["n_outside_3"] = df["n_outside_3"].astype("Int64")
    return df


def summarize(sp: pd.DataFrame, info: dict) -> dict:
    w = sp[sp.in_window]
    ok = w[w.all5_tracked]
    out = dict(clip=info["clip"], offense=info["offense"], defense=info["defense"],
               ends_in_shot=str(info["ends_in_shot"]).lower() == "true", result=info["result"],
               window_frames=len(w), frames_all5=len(ok),
               share_all5_tracked=round(len(ok) / max(len(w), 1), 3))
    for m in ("hull_area_sqft", "mean_dist_ft", "n_outside_3"):
        v = ok[m].astype(float)
        out[f"{m}_avg"] = round(float(v.mean()), 2) if len(v) else np.nan
        out[f"{m}_min"] = round(float(v.min()), 2) if len(v) else np.nan
    shot = info.get("shot_t_s")
    if out["ends_in_shot"] and shot == shot and shot is not None:
        last = ok[(ok.t_sec >= float(shot) - 1.0) & (ok.t_sec <= float(shot))]
        for m in ("hull_area_sqft", "mean_dist_ft", "n_outside_3"):
            out[f"{m}_last1s"] = round(float(last[m].astype(float).mean()), 2) if len(last) else np.nan
        out["last1s_frames"] = len(last)
    else:
        for m in ("hull_area_sqft", "mean_dist_ft", "n_outside_3"):
            out[f"{m}_last1s"] = np.nan
        out["last1s_frames"] = 0
    return out
