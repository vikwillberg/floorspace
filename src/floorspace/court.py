"""Court mapping: per-frame homography image <-> court feet.

Approach
--------
1. Line evidence per frame: thin bright, unsaturated ridges (white court lines)
   and thin dark purple ridges (the 3-point line), restricted to the floor
   area, with players and broadcast graphics masked out.
2. Camera model: the main broadcast camera pans/tilts/zooms from a fixed spot,
   so  H = K(f) * R(pan, tilt, roll) * [e1 e2 -C]  with one camera centre C for
   the whole game. Per frame only 4 numbers are fitted, which keeps the fit
   stable when few lines are visible and lets us smooth the camera path.
3. Keyframe anchor: a free 8-DOF homography is fitted on a frame with a clear
   paint (seeded from the four paint corners) and refined against all lines.
   Decomposing it gives f, R and C.
4. Tracking: frame-to-frame image motion from KLT feature tracks (players and
   graphics masked) predicts the next camera; the prediction is then refined
   against the line evidence of that frame, which removes drift.
5. The camera path (pan, tilt, roll, log f) is smoothed over time.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter

from . import config
from .courtmodel import court_lines, project, unproject

# ----------------------------------------------------------------------------
# Line evidence
# ----------------------------------------------------------------------------

_K15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


@dataclass
class Evidence:
    dt_white: np.ndarray      # distance (px) to nearest white-line pixel
    dt_purple: np.ndarray     # distance (px) to nearest purple-line pixel
    valid: np.ndarray         # bool: pixel can be used (floor, not player, not graphics)
    white: np.ndarray         # raw binary maps (for debugging)
    purple: np.ndarray
    scale: float              # evidence resolution relative to the frame


def player_mask(shape, boxes, pad=6):
    m = np.zeros(shape[:2], np.uint8)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(m, (int(x0) - pad, int(y0) - pad), (int(x1) + pad, int(y1) + pad), 255, -1)
    return m > 0


def overlay_mask(shape, overlay_boxes):
    m = np.zeros(shape[:2], bool)
    for x0, y0, x1, y1 in overlay_boxes:
        m[y0:y1, x0:x1] = True
    return m


def floor_color_mask(frame):
    """Wood, purple paint and the gold apron. Excludes the dark crowd."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    wood = (h <= 22) & (s >= 25) & (s <= 140) & (v >= 120)
    gold = (h >= 18) & (h <= 34) & (s >= 120) & (v >= 140)
    paint = (h >= 118) & (h <= 145) & (s >= 90) & (v >= 60)
    whiteish = (s < 60) & (v > 170)
    m = (wood | gold | paint | whiteish).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return m > 0


def court_region_mask(shape, H, margin_ft=3.0):
    if H is None:
        return np.ones(shape[:2], bool)
    poly = np.array([[-margin_ft, -margin_ft], [config.COURT_L + margin_ft, -margin_ft],
                     [config.COURT_L + margin_ft, config.COURT_W + margin_ft],
                     [-margin_ft, config.COURT_W + margin_ft]], float)
    # densify so the polygon stays valid when part of it is behind the camera
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        for t in np.linspace(0, 1, 30, endpoint=False):
            pts.append(a + t * (b - a))
    pts = np.array(pts)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    ok = p[:, 2] > 1e-6
    uv = (p[ok, :2] / p[ok, 2:3]).clip(-5000, 8000).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    if len(uv) >= 3:
        cv2.fillPoly(m, [cv2.convexHull(uv)], 255)
    return m > 0


def line_evidence(frame, boxes=(), overlay_boxes=(), H=None, scale=0.5) -> Evidence:
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(15 * scale * 2) | 1),) * 2)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hh, ss, vv = (hsv[:, :, i].astype(np.float32) for i in range(3))
    toph = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
    blackh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k)
    white = (toph > 16) & (ss < 75) & (vv > 140)
    purple = (blackh > 16) & (hh > 105) & (hh < 165) & (ss > 35)

    sh = small.shape
    Hs = None if H is None else np.diag([scale, scale, 1.0]) @ H
    valid = floor_color_mask(small) & court_region_mask(sh, Hs)
    if len(boxes):
        valid &= ~player_mask(sh, np.asarray(boxes) * scale, pad=int(8 * scale))
    if len(overlay_boxes):
        valid &= ~overlay_mask(sh, (np.asarray(overlay_boxes) * scale).astype(int))
    white &= valid
    purple &= valid
    dt_w = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, 5)
    dt_p = cv2.distanceTransform((~purple).astype(np.uint8), cv2.DIST_L2, 5)
    return Evidence(dt_w, dt_p, valid, white, purple, scale)


# ----------------------------------------------------------------------------
# Model points used for alignment
# ----------------------------------------------------------------------------

def _alignment_points():
    white, purple = [], []
    for kind, name, p in court_lines():
        if kind == "white":
            white.append(p)
        else:
            purple.append(p)
    return np.vstack(white), np.vstack(purple)


MODEL_WHITE, MODEL_PURPLE = _alignment_points()


def _sample(dt, uv, scale):
    """Bilinear sample a distance map at full-res pixel coords uv. Returns (d, inside)."""
    x, y = uv[:, 0] * scale, uv[:, 1] * scale
    h, w = dt.shape
    inside = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    d = map_coordinates(dt, [y.clip(0, h - 1), x.clip(0, w - 1)], order=1, mode="nearest")
    return d / scale, inside


def _valid_at(ev, uv):
    x = (uv[:, 0] * ev.scale).astype(int)
    y = (uv[:, 1] * ev.scale).astype(int)
    h, w = ev.valid.shape
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    ok = np.zeros(len(uv), bool)
    ok[inside] = ev.valid[y[inside], x[inside]]
    return ok


def alignment_residuals(H, ev, tau=12.0, purple_w=1.0, pts=None):
    """Truncated distances (px) from projected model points to line evidence."""
    res = []
    for model, dt, wgt in ((MODEL_WHITE, ev.dt_white, 1.0), (MODEL_PURPLE, ev.dt_purple, purple_w)):
        uv = project(H, model)
        fin = np.isfinite(uv).all(1)
        uv = np.where(fin[:, None], uv, -1e4)
        d, inside = _sample(dt, uv, ev.scale)
        use = inside & _valid_at(ev, uv)
        r = np.where(use, np.minimum(d, tau), 0.0) * wgt
        res.append(r)
    return np.concatenate(res)


def alignment_score(H, ev, tau=6.0):
    """Fraction of visible model points within tau px of evidence, and count visible."""
    hits = vis = 0
    for model, dt in ((MODEL_WHITE, ev.dt_white), (MODEL_PURPLE, ev.dt_purple)):
        uv = project(H, model)
        uv = np.where(np.isfinite(uv).all(1)[:, None], uv, -1e4)
        d, inside = _sample(dt, uv, ev.scale)
        use = inside & _valid_at(ev, uv)
        vis += use.sum()
        hits += (use & (d < tau)).sum()
    return hits / max(vis, 1), int(vis)


# ----------------------------------------------------------------------------
# Free homography refinement (keyframes)
# ----------------------------------------------------------------------------

def _H_from_corners(court_pts, img_pts):
    return cv2.getPerspectiveTransform(court_pts.astype(np.float32), img_pts.astype(np.float32))


def refine_free(H0, ev, court_box=(55.0, 3.0, 94.0, 47.0), taus=(40, 20, 10, 6)):
    """Refine an 8-DOF homography by moving the image positions of 4 court points."""
    x0, y0, x1, y1 = court_box
    cp = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    ip0 = project(H0, cp)
    H = H0
    for tau in taus:
        def fun(q):
            Hq = _H_from_corners(cp, q.reshape(4, 2))
            return alignment_residuals(Hq, ev, tau=tau)
        sol = least_squares(fun, ip0.ravel(), method="trf", loss="soft_l1", f_scale=tau / 3,
                            diff_step=1e-4, x_scale=10.0, max_nfev=400)
        ip0 = sol.x.reshape(4, 2)
        H = _H_from_corners(cp, ip0)
    return H


# ----------------------------------------------------------------------------
# PTZ camera model
# ----------------------------------------------------------------------------

@dataclass
class PTZ:
    pan: float    # radians, 0 = looking straight across the court (towards -y)
    tilt: float   # radians, >0 = looking down
    roll: float   # radians
    logf: float   # log focal length in pixels

    def as_array(self):
        return np.array([self.pan, self.tilt, self.roll, self.logf])


def rotation(pan, tilt, roll):
    """World (x right, y towards camera, z down) -> camera (x right, y down, z forward)."""
    cp_, sp_ = math.cos(pan), math.sin(pan)
    ct, st = math.cos(tilt), math.sin(tilt)
    f = np.array([sp_ * ct, -cp_ * ct, st])
    r = np.array([cp_, sp_, 0.0])
    u = np.cross(f, r)
    cr, sr = math.cos(roll), math.sin(roll)
    r2 = cr * r + sr * u
    u2 = -sr * r + cr * u
    return np.vstack([r2, u2, f])


def ptz_to_H(p, C, cx, cy):
    pan, tilt, roll, logf = p
    f = math.exp(logf)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    R = rotation(pan, tilt, roll)
    T = np.array([[1, 0, -C[0]], [0, 1, -C[1]], [0, 0, -C[2]]], float)
    H = K @ R @ T
    return H / H[2, 2]


def decompose(H, cx, cy):
    """Free homography -> (f, R, C) assuming square pixels and a centred principal point."""
    Hn = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ H
    h1, h2, h3 = Hn[:, 0], Hn[:, 1], Hn[:, 2]
    # r1.r2 = 0 with K^-1 = diag(1/f, 1/f, 1)
    num = -(h1[0] * h2[0] + h1[1] * h2[1])
    den = h1[2] * h2[2]
    f2_a = num / den if abs(den) > 1e-12 else np.nan
    # |r1| = |r2|
    den2 = h1[2] ** 2 - h2[2] ** 2
    f2_b = ((h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2)) / den2 if abs(den2) > 1e-12 else np.nan
    cands = [v for v in (f2_a, f2_b) if np.isfinite(v) and v > 0]
    f = math.sqrt(np.mean(cands))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    a1, a2, a3 = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = 2.0 / (np.linalg.norm(a1) + np.linalg.norm(a2))
    r1, r2 = a1 * lam, a2 * lam
    r3 = np.cross(r1, r2)
    t = a3 * lam
    if t[2] < 0 and False:
        pass
    R = np.c_[r1, r2, r3]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    # world->camera: X_cam = R_wc (X - C); here columns of R_wc are r1, r2, r3 and t = -R_wc C
    C = -R.T @ t
    if C[2] > 0:                      # camera must be above the floor (z down => negative)
        lam = -lam
        r1, r2, t = -r1, -r2, -t
        R = np.c_[r1, r2, np.cross(r1, r2)]
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        C = -R.T @ t
    return f, R, C


def R_to_ptz(R):
    """Inverse of rotation(): rows are r, u, f."""
    f = R[2]
    tilt = math.asin(np.clip(f[2], -1, 1))
    pan = math.atan2(f[0], -f[1])
    r0 = np.array([math.cos(pan), math.sin(pan), 0.0])
    u0 = np.cross(f, r0)
    roll = math.atan2(R[0] @ u0, R[0] @ r0)
    return pan, tilt, roll


def fit_ptz(p0, C, cx, cy, ev, taus=(24, 12, 6), prior=None, prior_w=None, max_nfev=200):
    p = np.asarray(p0, float)
    for tau in taus:
        def fun(q):
            r = alignment_residuals(ptz_to_H(q, C, cx, cy), ev, tau=tau)
            if prior is not None:
                r = np.concatenate([r, (q - prior) * prior_w])
            return r
        sol = least_squares(fun, p, method="trf", loss="soft_l1", f_scale=tau / 3,
                            diff_step=1e-6, x_scale=np.array([1e-3, 1e-3, 1e-3, 1e-3]),
                            max_nfev=max_nfev)
        p = sol.x
    return p


def H_to_ptz_given_C(H, C, cx, cy, p0, w, h):
    """Best PTZ (fixed C) reproducing a homography on a grid of court points."""
    gx, gy = np.meshgrid(np.linspace(0, config.COURT_L, 19), np.linspace(0, config.COURT_W, 11))
    pts = np.c_[gx.ravel(), gy.ravel()]
    uv = project(H, pts)
    ok = np.isfinite(uv).all(1) & (uv[:, 0] > -0.5 * w) & (uv[:, 0] < 1.5 * w) & \
        (uv[:, 1] > -0.5 * h) & (uv[:, 1] < 1.5 * h)
    pts, uv = pts[ok], uv[ok]

    def fun(q):
        return (project(ptz_to_H(q, C, cx, cy), pts) - uv).ravel()
    return least_squares(fun, np.asarray(p0, float), method="lm", x_scale=1e-3).x


# ----------------------------------------------------------------------------
# Frame-to-frame motion (KLT on static background, players/graphics masked)
# ----------------------------------------------------------------------------

def interframe_homography(prev_gray, gray, mask):
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1500, qualityLevel=0.005, minDistance=12,
                                  mask=mask.astype(np.uint8) * 255, blockSize=7)
    if pts is None or len(pts) < 30:
        return None, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=4,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, winSize=(21, 21), maxLevel=4)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < 1.0)
    if good.sum() < 30:
        return None, int(good.sum())
    M, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 2.0, maxIters=3000, confidence=0.999)
    if M is None:
        return None, 0
    return M, int(inl.sum())


# ----------------------------------------------------------------------------
# Full camera fit (shared camera centre over several keyframes)
# ----------------------------------------------------------------------------

def H_to_camera(H, cx, cy, w, h, x0=None, region=None):
    """Fit (C, pan, tilt, roll, logf) to a homography over the visible court region."""
    gx, gy = np.meshgrid(np.linspace(0, config.COURT_L, 48), np.linspace(0, config.COURT_W, 26))
    pts = np.c_[gx.ravel(), gy.ravel()]
    if region is not None:
        pts = pts[(pts[:, 0] >= region[0]) & (pts[:, 0] <= region[2])
                  & (pts[:, 1] >= region[1]) & (pts[:, 1] <= region[3])]
    uv = project(H, pts)
    ok = np.isfinite(uv).all(1) & (uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h)
    pts, uv = pts[ok], uv[ok]
    if x0 is None:
        f, R, C = decompose(H, cx, cy)
        x0 = np.r_[C, R_to_ptz(R), math.log(f)]

    def fun(q):
        return (project(ptz_to_H(q[3:], q[:3], cx, cy), pts) - uv).ravel()
    sol = least_squares(fun, x0, method="trf", loss="soft_l1", f_scale=5.0,
                        x_scale=np.r_[1, 1, 1, 1e-3, 1e-3, 1e-3, 1e-2])
    return sol.x, float(np.sqrt(np.mean(sol.fun ** 2)))


def fit_camera_multi(C0, ptz0_list, ev_list, cx, cy, taus=(24, 12, 6, 4)):
    """Jointly refine a shared camera centre and per-keyframe PTZ against line evidence."""
    n = len(ptz0_list)
    q = np.r_[np.asarray(C0, float), np.concatenate([np.asarray(p, float) for p in ptz0_list])]
    for tau in taus:
        def fun(v):
            C = v[:3]
            rs = []
            for i, ev in enumerate(ev_list):
                p = v[3 + 4 * i: 7 + 4 * i]
                rs.append(alignment_residuals(ptz_to_H(p, C, cx, cy), ev, tau=tau))
            return np.concatenate(rs)
        sol = least_squares(fun, q, method="trf", loss="soft_l1", f_scale=tau / 3, diff_step=1e-6,
                            x_scale=np.r_[1.0, 1.0, 1.0, np.tile([1e-3, 1e-3, 1e-3, 1e-3], n)],
                            max_nfev=300)
        q = sol.x
    return q[:3], [q[3 + 4 * i: 7 + 4 * i] for i in range(n)]


# ----------------------------------------------------------------------------
# Clip tracking
# ----------------------------------------------------------------------------

def _norm_gray(frame, scale=0.5):
    g = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6) * 48 + 128
    return g.clip(0, 255).astype(np.uint8)


def _static_mask(shape, boxes, overlay_boxes, scale=0.5):
    m = np.ones(shape[:2], bool)
    if len(boxes):
        m &= ~player_mask(shape, np.asarray(boxes) * scale, pad=int(10 * scale))
    if len(overlay_boxes):
        m &= ~overlay_mask(shape, (np.asarray(overlay_boxes) * scale).astype(int))
    return m


def ptz_prior_weights(p, sigma_px=4.0, w=2360):
    f = math.exp(p[3])
    return np.array([f, f, f, w / 2]) / sigma_px


def track_camera(frames, boxes_per_frame, overlay_boxes, C, anchor_idx, anchor_ptz, cx, cy,
                 log=print):
    """Propagate the PTZ camera through a clip from an anchor frame (forward + backward)."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    S = np.diag([0.5, 0.5, 1.0])
    Sinv = np.linalg.inv(S)
    grays = [_norm_gray(f) for f in frames]
    raw = np.full((n, 4), np.nan)
    score = np.zeros(n)
    nvis = np.zeros(n, int)
    inliers = np.zeros(n, int)
    raw[anchor_idx] = anchor_ptz
    ev = line_evidence(frames[anchor_idx], boxes_per_frame[anchor_idx], overlay_boxes,
                       ptz_to_H(anchor_ptz, C, cx, cy))
    score[anchor_idx], nvis[anchor_idx] = alignment_score(ptz_to_H(anchor_ptz, C, cx, cy), ev)

    for direction in (+1, -1):
        rng = range(anchor_idx + 1, n) if direction > 0 else range(anchor_idx - 1, -1, -1)
        for t in rng:
            s = t - direction
            p_prev = raw[s]
            H_prev = ptz_to_H(p_prev, C, cx, cy)
            mask = _static_mask(grays[s].shape, boxes_per_frame[s], overlay_boxes)
            M, ninl = interframe_homography(grays[s], grays[t], mask)
            inliers[t] = ninl
            if M is not None:
                H_pred = Sinv @ M @ S @ H_prev
                p_pred = H_to_ptz_given_C(H_pred, C, cx, cy, p_prev, w, h)
            else:
                p_pred = p_prev.copy()
            ev = line_evidence(frames[t], boxes_per_frame[t], overlay_boxes, ptz_to_H(p_pred, C, cx, cy))
            p_fit = fit_ptz(p_pred, C, cx, cy, ev, taus=(12, 6),
                            prior=p_pred, prior_w=ptz_prior_weights(p_pred))
            raw[t] = p_fit
            score[t], nvis[t] = alignment_score(ptz_to_H(p_fit, C, cx, cy), ev)
            if t % 20 == 0:
                log(f"    cam t={t:3d} score={score[t]:.2f} vis={nvis[t]} klt_inliers={ninl}")
    return raw, score, nvis, inliers


def smooth_ptz(raw, score, window=9, poly=2):
    """Savitzky-Golay smoothing of the camera path (pan, tilt, roll, log f)."""
    out = raw.copy()
    n = len(raw)
    if n < 5:
        return out
    win = min(window, n if n % 2 == 1 else n - 1)
    for k in range(4):
        out[:, k] = savgol_filter(raw[:, k], win, min(poly, win - 1), mode="interp")
    return out


# ----------------------------------------------------------------------------
# Automatic initialisation (no seed): coarse search over pan / tilt / zoom
# ----------------------------------------------------------------------------

def camera_P(p, C, cx, cy):
    """3x4 projection matrix (world feet, z down) for PTZ params p."""
    pan, tilt, roll, logf = p
    f = math.exp(logf)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    R = rotation(pan, tilt, roll)
    return K @ R @ np.c_[np.eye(3), -np.asarray(C, float)]


def paint_mask(frame, scale=0.25):
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] >= 118) & (hsv[:, :, 0] <= 145) & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 50))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m > 0


_PAINTS = np.array([[[0, 17], [19, 17], [19, 33], [0, 33]],
                    [[75, 17], [94, 17], [94, 33], [75, 33]]], float)


def paint_iou(p, C, cx, cy, pm, dontcare, scale):
    H = ptz_to_H(p, C, cx, cy)
    q = np.zeros(pm.shape, np.uint8)
    for poly in _PAINTS:
        pp = np.c_[poly, np.ones(4)] @ H.T
        if (pp[:, 2] <= 0).any():
            continue
        uv = pp[:, :2] / pp[:, 2:3] * scale
        if np.abs(uv).max() > 1e5:
            continue
        cv2.fillPoly(q, [np.round(uv).astype(np.int32)], 1)
    q = q > 0
    inter = (q & pm & ~dontcare).sum()
    union = ((q | pm) & ~dontcare).sum()
    return inter / max(union, 1)


def paint_search(frame, boxes, overlay_boxes, C, cx, cy, roll=0.0, scale=0.25, log=print):
    """Coarse-to-fine search for the PTZ whose projected paints best overlap the purple paint."""
    pm = paint_mask(frame, scale)
    dc = np.zeros(pm.shape, bool)
    if len(boxes):
        dc |= player_mask(pm.shape, np.asarray(boxes) * scale, pad=1)
    if len(overlay_boxes):
        dc |= overlay_mask(pm.shape, (np.asarray(overlay_boxes) * scale).astype(int))
    pm &= ~dc
    best = []
    for pan in np.linspace(-0.5, 0.5, 51):
        for tilt in np.linspace(0.14, 0.40, 14):
            for logf in np.log(np.geomspace(2800, 10000, 14)):
                p = np.array([pan, tilt, roll, logf])
                best.append((paint_iou(p, C, cx, cy, pm, dc, scale), p))
    best.sort(key=lambda b: -b[0])
    refined = []
    for iou0, p0 in best[:8]:
        steps = np.array([0.02, 0.02, 0.0, 0.09])
        p, iou = p0.copy(), iou0
        for _ in range(60):                  # greedy coordinate search with shrinking steps
            improved = False
            for k in (0, 1, 3):
                for sgn in (+1, -1):
                    q = p.copy()
                    q[k] += sgn * steps[k]
                    v = paint_iou(q, C, cx, cy, pm, dc, scale)
                    if v > iou:
                        p, iou, improved = q, v, True
            if not improved:
                steps /= 2
                if steps[0] < 1e-4:
                    break
        refined.append((iou, p))
    refined.sort(key=lambda r: -r[0])
    log(f"    paint IoU candidates: " + ", ".join(f"{r[0]:.2f}@pan{r[1][0]:+.3f}" for r in refined[:4]))
    return refined[0][1], refined[0][0]


def auto_anchor(frame, boxes, overlay_boxes, C, cx, cy, roll=0.0, log=print):
    """Find the camera for one frame without any seed: paint overlap, then line refinement."""
    p, iou = paint_search(frame, boxes, overlay_boxes, C, cx, cy, roll=roll, log=log)
    for taus in ((40, 20, 10), (8, 4)):
        ev = line_evidence(frame, boxes, overlay_boxes, ptz_to_H(p, C, cx, cy))
        p = fit_ptz(p, C, cx, cy, ev, taus=taus)
    ev = line_evidence(frame, boxes, overlay_boxes, ptz_to_H(p, C, cx, cy))
    sc = alignment_score(ptz_to_H(p, C, cx, cy), ev)
    return p, iou, sc


# ----------------------------------------------------------------------------
# Pipeline entry points
# ----------------------------------------------------------------------------

CAMERA_JSON = config.DATA_DIR / "camera.json"
DEFAULT_C = np.array([47.0, 135.0, -32.0])   # generic broadcast position, refined by calibrate()


def load_camera():
    if CAMERA_JSON.exists():
        j = json.loads(CAMERA_JSON.read_text())
        return np.array(j["C"]), float(j["roll"])
    return DEFAULT_C.copy(), 0.0


def read_frames(clip):
    cap = cv2.VideoCapture(str(config.MP4_DIR / f"{clip}.mp4"))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def person_boxes(clip, n, conf=0.15):
    from .detect_track import detect, PERSON
    d = detect(clip)
    return [d["xyxy"][(d["frame"] == i) & (d["cls"] == PERSON) & (d["conf"] > conf)] for i in range(n)]


def calibrate(keyframes, log=print):
    """Estimate the shared camera centre from keyframes [(clip, frame), ...] of several clips."""
    from .prepare import load_meta
    C, roll = DEFAULT_C.copy(), 0.0
    for it in range(2):
        ptz, evs = [], []
        for clip, fi in keyframes:
            frames = read_frames(clip)
            fr = frames[fi]
            boxes = person_boxes(clip, len(frames))[fi]
            ov = load_meta(clip)["overlay_mask_boxes"]
            cx, cy = fr.shape[1] / 2, fr.shape[0] / 2
            p, iou, sc = auto_anchor(fr, boxes, ov, C, cx, cy, roll=roll, log=log)
            log(f"  calib iter {it} {clip} f{fi}: line score {sc[0]:.2f}")
            ptz.append(p)
            evs.append(line_evidence(fr, boxes, ov, ptz_to_H(p, C, cx, cy)))
        C, ptz = fit_camera_multi(C, ptz, evs, cx, cy)
        roll = float(np.median([p[2] for p in ptz]))
        log(f"  calib iter {it}: C = {np.round(C, 2)}  roll = {roll:.4f}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    CAMERA_JSON.write_text(json.dumps({
        "C": [round(float(v), 4) for v in C], "roll": round(roll, 5),
        "note": "Main broadcast camera centre in court feet (x along court, y towards camera, z down). "
                "Estimated jointly from keyframes by `python -m floorspace calibrate`.",
        "keyframes": [[c, int(f)] for c, f in keyframes]}, indent=2))
    return C, roll


def map_clip(clip, force=False, log=print):
    """Per-frame camera for a clip -> work/<clip>/camera.npz (H, P, ptz, score)."""
    from .prepare import load_meta
    out = config.WORK_DIR / clip / "camera.npz"
    if out.exists() and not force:
        return dict(np.load(out))
    C, roll = load_camera()
    frames = read_frames(clip)
    n = len(frames)
    h, w = frames[0].shape[:2]
    cx, cy = w / 2, h / 2
    boxes = person_boxes(clip, n)
    ov = load_meta(clip)["overlay_mask_boxes"]
    # anchor: best of a few candidate frames
    best = None
    for fi in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}):
        p, iou, sc = auto_anchor(frames[fi], boxes[fi], ov, C, cx, cy, roll=roll, log=log)
        log(f"  {clip} anchor candidate f{fi}: paint IoU {iou:.2f}, line score {sc[0]:.2f} ({sc[1]} pts)")
        if sc[1] >= 150 and (best is None or sc[0] > best[2]):
            best = (fi, p, sc[0])
    fi, p, sc = best
    log(f"  {clip} anchor = frame {fi} (line score {sc:.2f})")
    raw, score, nvis, inl = track_camera(frames, boxes, ov, C, fi, p, cx, cy, log=log)
    # re-anchor stretches where the line fit degraded
    bad = np.flatnonzero(score < 0.5)
    if len(bad):
        log(f"  {clip}: {len(bad)} frames with line score < 0.5 -> re-anchoring")
        for t in bad:
            if score[t] >= 0.5:
                continue
            p2, iou2, sc2 = auto_anchor(frames[t], boxes[t], ov, C, cx, cy, roll=roll, log=lambda *a: None)
            if sc2[0] > score[t] + 0.1:
                raw[t], score[t], nvis[t] = p2, sc2[0], sc2[1]
    sm = smooth_ptz(raw, score)
    Hs = np.stack([ptz_to_H(q, C, cx, cy) for q in sm])
    Ps = np.stack([camera_P(q, C, cx, cy) for q in sm])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, raw=raw, sm=sm, score=score, nvis=nvis, inliers=inl, H=Hs, P=Ps, C=C,
             anchor=fi, w=w, h=h)
    log(f"  {clip}: camera mapped, line score median {np.median(score):.2f}, min {score.min():.2f}")
    return dict(np.load(out))
