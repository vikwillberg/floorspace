"""Person + ball detection (Ultralytics YOLO) and multi-object tracking (ByteTrack).

Detection is the slow step on CPU, so raw detections are cached per clip in
work/<clip>/detections.npz and tracking runs separately on the cache. That
keeps tracker tuning cheap.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import cv2
import numpy as np

from . import config

PERSON, BALL = 0, 32
DET_MODEL = "yolo11x.pt"
DET_IMGSZ = 1536
DET_CONF = 0.05


def iter_frames(clip: str):
    cap = cv2.VideoCapture(str(config.MP4_DIR / f"{clip}.mp4"))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield i, fr
        i += 1
    cap.release()


def n_frames(clip: str) -> int:
    cap = cv2.VideoCapture(str(config.MP4_DIR / f"{clip}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def detect(clip: str, force: bool = False) -> np.lib.npyio.NpzFile:
    out = config.WORK_DIR / clip / "detections.npz"
    if out.exists() and not force:
        return np.load(out)
    from ultralytics import YOLO
    out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(config.WEIGHTS_DIR / DET_MODEL))
    rows = []
    t0 = time.time()
    total = n_frames(clip)
    for i, fr in iter_frames(clip):
        r = model.predict(fr, imgsz=DET_IMGSZ, classes=[PERSON, BALL], conf=DET_CONF,
                          iou=0.7, verbose=False)[0]
        b = r.boxes
        xyxy, conf, cls = b.xyxy.numpy(), b.conf.numpy(), b.cls.numpy()
        for k in range(len(conf)):
            rows.append((i, *xyxy[k], conf[k], cls[k]))
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  {clip} det frame {i}/{total}  {el / (i + 1):.2f}s/frame", flush=True)
    a = np.array(rows, dtype=np.float32).reshape(-1, 7)
    np.savez_compressed(out, frame=a[:, 0].astype(np.int32), xyxy=a[:, 1:5], conf=a[:, 5],
                        cls=a[:, 6].astype(np.int32), n_frames=total, model=DET_MODEL,
                        imgsz=DET_IMGSZ)
    return np.load(out)


# ----------------------------------------------------------------------------
# Court filter + ByteTrack with camera-motion compensation
# ----------------------------------------------------------------------------

class _Dets:
    """Minimal numpy 'Boxes'-like container accepted by Ultralytics' BYTETracker."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.asarray(cls, np.float32).reshape(-1)

    @property
    def xywh(self):
        x0, y0, x1, y1 = self.xyxy.T
        return np.c_[(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


def _make_tracker(**over):
    from ultralytics.trackers.byte_tracker import BYTETracker

    class CMCByteTracker(BYTETracker):
        """ByteTrack whose Kalman predictions are moved with the camera before matching."""
        warp = None  # 3x3 image homography prev -> current, set before each update()

        def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high):
            M = self.warp
            if M is None:
                return
            for st in list(strack_pool) + list(unconfirmed):
                m, P = st.mean.copy(), st.covariance
                x, y = m[0], m[1]
                v = M @ np.array([x, y, 1.0])
                nx, ny = v[0] / v[2], v[1] / v[2]
                # local Jacobian of the homography at (x, y)
                eps = 1.0
                vx = M @ np.array([x + eps, y, 1.0])
                vy = M @ np.array([x, y + eps, 1.0])
                J = np.array([[vx[0] / vx[2] - nx, vy[0] / vy[2] - nx],
                              [vx[1] / vx[2] - ny, vy[1] / vy[2] - ny]]) / eps
                s = math.sqrt(abs(np.linalg.det(J)))
                T = np.eye(8)
                T[0:2, 0:2] = J
                T[3, 3] = s
                T[4:6, 4:6] = J
                T[7, 7] = s
                m = T @ m
                m[0], m[1] = nx, ny
                st.mean = m
                st.covariance = T @ P @ T.T

    cfg = dict(track_high_thresh=0.30, track_low_thresh=0.10, new_track_thresh=0.35,
               track_buffer=45, match_thresh=0.85, fuse_score=True)
    cfg.update(over)
    return CMCByteTracker(SimpleNamespace(**cfg))


def foot_points(xyxy):
    return np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]]


def court_filter(xyxy, H, P, margin_x=2.0, margin_y=2.0, ratio=(0.62, 1.8), person_ft=6.5):
    """Keep detections whose feet are on the court and whose size fits a standing person."""
    from .courtmodel import unproject
    if len(xyxy) == 0:
        return np.zeros(0, bool), np.zeros((0, 2))
    ft = unproject(H, foot_points(xyxy))
    inside = ((ft[:, 0] > -margin_x) & (ft[:, 0] < config.COURT_L + margin_x)
              & (ft[:, 1] > -margin_y) & (ft[:, 1] < config.COURT_W + margin_y))
    # expected pixel height of a 6.5 ft person standing at that spot (world z is down)
    X = np.c_[ft, np.zeros(len(ft)), np.ones(len(ft))]
    Xh = np.c_[ft, -person_ft * np.ones(len(ft)), np.ones(len(ft))]
    a, b = X @ P.T, Xh @ P.T
    exp_h = np.abs(a[:, 1] / a[:, 2] - b[:, 1] / b[:, 2])
    r = (xyxy[:, 3] - xyxy[:, 1]) / np.maximum(exp_h, 1e-6)
    return inside & (r > ratio[0]) & (r < ratio[1]), ft


def track_clip(clip: str, Hs: np.ndarray, Ps: np.ndarray, overlay_boxes) -> "pd.DataFrame":
    """Run ByteTrack (with camera motion compensation) on cached, court-filtered detections."""
    import pandas as pd
    d = detect(clip)
    fr_idx, xyxy, conf, cls = d["frame"], d["xyxy"], d["conf"], d["cls"]
    n = int(d["n_frames"])
    tracker = _make_tracker()
    rows = []
    ob = np.asarray(overlay_boxes)
    for t in range(n):
        sel = (fr_idx == t) & (cls == PERSON)
        bx, cf = xyxy[sel], conf[sel]
        keep, ft = court_filter(bx, Hs[t], Ps[t])
        # drop boxes mostly inside a broadcast graphic
        if len(bx) and len(ob):
            for (gx0, gy0, gx1, gy1) in ob:
                ix = np.clip(np.minimum(bx[:, 2], gx1) - np.maximum(bx[:, 0], gx0), 0, None)
                iy = np.clip(np.minimum(bx[:, 3], gy1) - np.maximum(bx[:, 1], gy0), 0, None)
                area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
                keep &= (ix * iy) / np.maximum(area, 1) < 0.5
        bx, cf = bx[keep], cf[keep]
        tracker.warp = None if t == 0 else Hs[t] @ np.linalg.inv(Hs[t - 1])
        out = tracker.update(_Dets(bx, cf, np.zeros(len(cf))), None)
        for x0, y0, x1, y1, tid, sc, _c, idx in out:
            rows.append((t, int(tid), x0, y0, x1, y1, sc, int(idx)))
    return pd.DataFrame(rows, columns=["frame", "track_id", "x0", "y0", "x1", "y1", "score", "det_idx"])


# ----------------------------------------------------------------------------
# Player cut-out masks (so floor graphics can go *under* the players)
# ----------------------------------------------------------------------------

SEG_MODEL = "yolo11l-seg.pt"
SEG_IMGSZ = 1280


def player_masks(clip: str, Hs, Ps, force=False, log=print) -> np.ndarray:
    """Per-frame union mask of people standing on the court (players + referees).

    Stored bit-packed in work/<clip>/player_masks.npz at full resolution.
    """
    out = config.WORK_DIR / clip / "player_masks.npz"
    if out.exists() and not force:
        z = np.load(out)
        return np.unpackbits(z["bits"], axis=-1, count=int(z["w"])).astype(bool)
    from ultralytics import YOLO
    model = YOLO(str(config.WEIGHTS_DIR / SEG_MODEL))
    masks = []
    t0 = time.time()
    for i, fr in iter_frames(clip):
        r = model.predict(fr, imgsz=SEG_IMGSZ, classes=[PERSON], conf=0.15, retina_masks=True,
                          verbose=False)[0]
        h, w = fr.shape[:2]
        m = np.zeros((h, w), bool)
        if r.masks is not None and len(r.boxes):
            keep, _ = court_filter(r.boxes.xyxy.numpy(), Hs[i], Ps[i])
            md = r.masks.data.numpy() > 0.5
            for k in np.flatnonzero(keep):
                m |= md[k]
        masks.append(m)
        if i % 25 == 0:
            log(f"  {clip} seg frame {i}  {(time.time() - t0) / (i + 1):.2f}s/frame")
    masks = np.stack(masks)
    np.savez_compressed(out, bits=np.packbits(masks, axis=-1), w=masks.shape[2])
    return masks
