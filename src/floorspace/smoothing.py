"""Smooth player trajectories in court coordinates.

* constant-velocity Kalman filter + Rauch-Tung-Striebel smoother (feet),
* short gaps are bridged by the smoother (state "interpolated"),
* after the last sighting a player is held for HOLD_FRAMES and then faded out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FPS = 30.0
MAX_BRIDGE = 24      # frames: gaps up to 0.8 s are bridged
HOLD_FRAMES = 12     # frames: hold a lost player for 0.4 s ...
FADE_FRAMES = 10     # ... then fade him out over 0.33 s
SIGMA_MEAS = 0.7     # ft, foot-point noise
SIGMA_ACC = 14.0     # ft/s^2, process noise (players accelerate hard)


def rts_smooth(t_obs, z, n_frames, f_start, f_end):
    """RTS smoother for one track on frames f_start..f_end. z: (k, 2) observations at t_obs."""
    dt = 1.0 / FPS
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = SIGMA_ACC ** 2
    G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    Q = G @ G.T * q
    Hm = np.zeros((2, 4))
    Hm[0, 0] = Hm[1, 1] = 1
    R = np.eye(2) * SIGMA_MEAS ** 2
    T = f_end - f_start + 1
    obs = {int(t): zz for t, zz in zip(t_obs, z)}
    x = np.r_[z[0], 0, 0]
    P = np.diag([1.0, 1.0, 25.0, 25.0])
    xs_f, Ps_f, xs_p, Ps_p = [], [], [], []
    for k in range(T):
        f = f_start + k
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xs_p.append(x.copy())
        Ps_p.append(P.copy())
        if f in obs:
            y = obs[f] - Hm @ x
            S = Hm @ P @ Hm.T + R
            # robust: inflate R for outliers (> 3 sigma)
            md = float(y @ np.linalg.solve(S, y))
            if md > 16:
                S = Hm @ P @ Hm.T + R * (md / 4)
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ Hm) @ P
        xs_f.append(x.copy())
        Ps_f.append(P.copy())
    xs = [None] * T
    xs[-1] = xs_f[-1]
    Ps = Ps_f[-1]
    for k in range(T - 2, -1, -1):
        C = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
    return np.array(xs)


def smooth_tracks(ids: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """ids: rows (frame, pid, team, cx, cy) -> dense per-frame table with state + alpha."""
    out = []
    for pid, g in ids.groupby("pid"):
        g = g.sort_values("frame")
        fr = g.frame.to_numpy()
        z = g[["cx", "cy"]].to_numpy(float)
        team = g.team.iloc[0]
        # split into runs where the gap is bridgeable
        breaks = np.flatnonzero(np.diff(fr) > MAX_BRIDGE)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(fr) - 1]
        for s, e in zip(starts, ends):
            f0, f1 = int(fr[s]), int(fr[e])
            f_end = min(n_frames - 1, f1 + HOLD_FRAMES + FADE_FRAMES)
            xs = rts_smooth(fr[s:e + 1], z[s:e + 1], n_frames, f0, f1)
            seen = set(fr[s:e + 1].tolist())
            for k in range(f0, f_end + 1):
                if k <= f1:
                    x, y = xs[k - f0, :2]
                    state = "observed" if k in seen else "interpolated"
                    alpha = 1.0
                else:  # hold at last smoothed position, then fade
                    x, y = xs[-1, :2]
                    lost = k - f1
                    state = "held" if lost <= HOLD_FRAMES else "fading"
                    alpha = 1.0 if lost <= HOLD_FRAMES else max(0.0, 1 - (lost - HOLD_FRAMES) / FADE_FRAMES)
                # fade in over a few frames when a track appears mid-clip
                if f0 > 0 and k - f0 < 5:
                    alpha *= (k - f0 + 1) / 5
                out.append((k, pid, team, x, y, state, alpha))
    df = pd.DataFrame(out, columns=["frame", "pid", "team", "x_ft", "y_ft", "state", "alpha"])
    # one row per pid per frame (a later run beats a fading earlier run)
    rank = df.state.map({"observed": 0, "interpolated": 1, "held": 2, "fading": 3})
    df = df.assign(_r=rank).sort_values(["frame", "pid", "_r"]).drop_duplicates(["frame", "pid"])
    return df.drop(columns="_r").reset_index(drop=True)


def cap_per_team(df: pd.DataFrame, max_per_team=5) -> pd.DataFrame:
    """Never show more than 5 players of a team: drop held/fading players first."""
    rank = df.state.map({"observed": 0, "interpolated": 1, "held": 2, "fading": 3})
    df = df.assign(_r=rank).sort_values(["frame", "team", "_r", "pid"])
    df = df.groupby(["frame", "team"], group_keys=False).head(max_per_team)
    return df.drop(columns="_r").sort_values(["frame", "pid"]).reset_index(drop=True)
