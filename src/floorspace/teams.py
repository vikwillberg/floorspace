"""Team assignment from jersey colours + identity clean-up.

Features (per detection): colour statistics of the torso, shorts and lower
legs. The Lakers wear gold, the Heat black, referees grey shirts with black
trousers. Features are averaged per track and clustered (k-means, k=3) over
all clips together; clusters are named by their colour statistics:
  * highest gold fraction on the torso           -> LAL
  * of the rest, darkest lower legs (trousers)   -> REF
  * remaining                                     -> MIA
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

REGIONS = {"torso": (0.16, 0.44, 0.28, 0.72), "shorts": (0.44, 0.60, 0.25, 0.75),
           "legs": (0.66, 0.88, 0.30, 0.70)}


def _region(frame, box, r):
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    a, b, c, d = r
    ys, ye = int(y0 + a * h), int(y0 + b * h)
    xs, xe = int(x0 + c * w), int(x0 + d * w)
    ys, xs = max(ys, 0), max(xs, 0)
    ye, xe = min(ye, frame.shape[0]), min(xe, frame.shape[1])
    if ye - ys < 3 or xe - xs < 3:
        return None
    return frame[ys:ye, xs:xe]


def region_stats(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv.T
    gold = ((h >= 14) & (h <= 34) & (s >= 100) & (v >= 110)).mean()
    dark = (v < 70).mean()
    light = ((s < 45) & (v > 135)).mean()
    red = (((h <= 8) | (h >= 170)) & (s >= 110) & (v >= 70)).mean()
    return np.r_[gold, dark, light, red, np.median(lab, axis=0) / 255.0]


# ----------------------------------------------------------------------------
# Floor background model ("clean plate"): lets us ignore floor pixels inside a box,
# e.g. the gold apron behind a Heat player standing on the baseline.
# ----------------------------------------------------------------------------

TEX_PPF = 6.0                      # texture pixels per foot
TEX_X0, TEX_Y0 = -12.0, -10.0      # texture covers the court plus the apron
TEX_W, TEX_H = int(118 * TEX_PPF), int(70 * TEX_PPF)
_T = np.array([[1 / TEX_PPF, 0, TEX_X0], [0, 1 / TEX_PPF, TEX_Y0], [0, 0, 1.0]])


def build_court_texture(frames, Hs, boxes_per_frame, step=3):
    """Median top-down image of the floor with players masked out."""
    stack = []
    for t in range(0, len(frames), step):
        M = Hs[t] @ _T
        tex = cv2.warpPerspective(frames[t], M, (TEX_W, TEX_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        h, w = frames[t].shape[:2]
        m = np.full((h, w), 255, np.uint8)
        for x0, y0, x1, y1 in boxes_per_frame[t]:
            cv2.rectangle(m, (int(x0) - 8, int(y0) - 8), (int(x1) + 8, int(y1) + 30), 0, -1)
        mw = cv2.warpPerspective(m, M, (TEX_W, TEX_H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tex[mw == 0] = np.nan
        stack.append(tex)
    tex = np.nanmedian(np.stack(stack), axis=0)
    valid = ~np.isnan(tex).any(2)
    return np.nan_to_num(tex).astype(np.uint8), valid


def clean_plate(texture, valid, H, shape):
    M = np.linalg.inv(H @ _T)
    h, w = shape[:2]
    plate = cv2.warpPerspective(texture, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pv = cv2.warpPerspective(valid.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return plate, pv > 0


def detection_features(frame, box, plate=None, plate_valid=None, de_thr=16.0):
    out = []
    for r in REGIONS.values():
        p = _region(frame, box, r)
        if p is None:
            out.append(np.full(8, np.nan))
            continue
        keep = np.ones(p.shape[:2], bool)
        if plate is not None:
            x0, y0, x1, y1 = box
            h, w = y1 - y0, x1 - x0
            ys, xs = max(int(y0 + r[0] * h), 0), max(int(x0 + r[2] * w), 0)
            bp = plate[ys:ys + p.shape[0], xs:xs + p.shape[1]]
            bv = plate_valid[ys:ys + p.shape[0], xs:xs + p.shape[1]]
            la = cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32)
            lb = cv2.cvtColor(bp, cv2.COLOR_BGR2LAB).astype(np.float32)
            keep = ~(bv & (np.linalg.norm(la - lb, axis=2) < de_thr))
        frac = keep.mean()
        if frac < 0.2:
            out.append(np.r_[np.full(7, np.nan), frac])
            continue
        out.append(np.r_[region_stats(p[keep][:, None, :]), frac])
    return np.concatenate(out)


FEATURE_NAMES = [f"{r}_{k}" for r in REGIONS for k in ("gold", "dark", "light", "red", "L", "a", "b", "fg")]


def features_for_tracks(frames, tracks: pd.DataFrame, Hs=None, texture=None, every=1) -> pd.DataFrame:
    rows = []
    for t, g in tracks[tracks.frame % every == 0].groupby("frame"):
        fr = frames[t]
        plate = pv = None
        if texture is not None:
            plate, pv = clean_plate(texture[0], texture[1], Hs[t], fr.shape)
        for r in g.itertuples():
            rows.append((r.track_id, t, *detection_features(fr, (r.x0, r.y0, r.x1, r.y1), plate, pv)))
    return pd.DataFrame(rows, columns=["track_id", "frame", *FEATURE_NAMES])


def name_clusters(centers: np.ndarray) -> dict[int, str]:
    gi = FEATURE_NAMES.index("torso_gold")
    li = FEATURE_NAMES.index("legs_dark")
    order = list(range(len(centers)))
    lal = int(np.argmax(centers[:, gi]))
    rest = [i for i in order if i != lal]
    ref = max(rest, key=lambda i: centers[i, li])
    mia = [i for i in rest if i != ref][0]
    return {lal: "LAL", ref: "REF", mia: "MIA"}


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------

CLUSTER_FEATURES = ["upper_dark", "upper_gold", "legs_dark"]


def cluster_space(F: pd.DataFrame) -> np.ndarray:
    """3-D colour space: Heat black (upper body), Lakers gold (upper body), referee trousers."""
    up_dark = (F.torso_dark + F.shorts_dark) / 2
    up_gold = (F.torso_gold + F.shorts_gold) / 2
    return np.c_[up_dark, up_gold, F.legs_dark]


def smooth_features(F: pd.DataFrame, win=15) -> pd.DataFrame:
    """Rolling median of the colour features along each ByteTrack track (cuts occlusion noise)."""
    F = F.sort_values(["track_id", "frame"]).copy()
    X = cluster_space(F)
    for j, n in enumerate(CLUSTER_FEATURES):
        F[n] = X[:, j]
        F[n + "_s"] = F.groupby("track_id")[n].transform(
            lambda v: v.rolling(win, center=True, min_periods=1).median())
    return F


def fit_clusters(F: pd.DataFrame, seed=0):
    """Two-stage k-means: (1) gold vs dark upper body, (2) Heat vs referees by trouser darkness."""
    from sklearn.cluster import KMeans
    S = F[[n + "_s" for n in CLUSTER_FEATURES]].dropna().to_numpy()
    km1 = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(S[:, :2])
    gold = int(np.argmax(km1.cluster_centers_[:, 1]))
    dark_rows = S[km1.labels_ != gold]
    km2 = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(dark_rows[:, 2:3])
    ref = int(np.argmax(km2.cluster_centers_[:, 0]))
    return {"stage1_centers": km1.cluster_centers_.tolist(), "stage1_gold": gold,
            "stage2_centers": km2.cluster_centers_.ravel().tolist(), "stage2_ref": ref,
            "features": CLUSTER_FEATURES, "n_rows": int(len(S))}


def label_rows(F: pd.DataFrame, model) -> np.ndarray:
    S = F[[n + "_s" for n in CLUSTER_FEATURES]].to_numpy()
    c1 = np.array(model["stage1_centers"])
    d1 = np.linalg.norm(np.nan_to_num(S[:, None, :2], nan=9) - c1[None], axis=2)
    is_gold = np.argmin(d1, axis=1) == model["stage1_gold"]
    c2 = np.array(model["stage2_centers"])
    is_ref = np.argmin(np.abs(np.nan_to_num(S[:, 2:3], nan=9) - c2[None]), axis=1) == model["stage2_ref"]
    lab = np.where(is_gold, "LAL", np.where(is_ref, "REF", "MIA")).astype(object)
    lab[np.isnan(S).any(1)] = "UNK"
    return lab


# ----------------------------------------------------------------------------
# Identity clean-up: split on team changes, re-link per team by continuity
# ----------------------------------------------------------------------------

def _rolling_majority(labels, win=9):
    labels = np.asarray(labels, dtype=object)
    out = labels.copy()
    half = win // 2
    for i in range(len(labels)):
        w = labels[max(0, i - half): i + half + 1]
        w = w[w != "UNK"]
        if len(w):
            vals, cnt = np.unique(w, return_counts=True)
            out[i] = vals[np.argmax(cnt)]
    return out


def split_segments(tr: pd.DataFrame, min_len=4) -> pd.DataFrame:
    """Split each ByteTrack track wherever its (smoothed) team label changes."""
    tr = tr.sort_values(["track_id", "frame"]).copy()
    seg_ids = np.zeros(len(tr), int)
    seg_team = np.empty(len(tr), dtype=object)
    sid = 0
    pos = 0
    for tid, g in tr.groupby("track_id", sort=False):
        sm = _rolling_majority(g.team_row.to_numpy())
        start = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or sm[i] != sm[start] or g.frame.iloc[i] - g.frame.iloc[i - 1] > 15:
                seg_ids[pos + start: pos + i] = sid
                seg_team[pos + start: pos + i] = sm[start]
                sid += 1
                start = i
        pos += len(g)
    tr["seg"] = seg_ids
    tr["team"] = seg_team
    n = tr.groupby("seg").frame.transform("count")
    return tr[n >= min_len]


def link_segments(tr: pd.DataFrame, max_gap=75, max_overlap=10, max_speed=26.0, base_ft=4.0,
                  new_cost=12.0, overlap_ft=3.0):
    """Link segments of one team into identities (Hungarian on segment ends vs starts).

    Two segments may also overlap by a few frames (a duplicate box on the same
    player while ByteTrack hands over) if they agree on position meanwhile.
    """
    from scipy.optimize import linear_sum_assignment
    segs = []
    by_seg = {}
    for sid, g in tr.groupby("seg"):
        g = g.sort_values("frame")
        by_seg[sid] = g.set_index("frame")[["cx", "cy"]]
        segs.append(dict(seg=sid, f0=int(g.frame.iloc[0]), f1=int(g.frame.iloc[-1]),
                         p0=g[["cx", "cy"]].iloc[:3].median().to_numpy(),
                         p1=g[["cx", "cy"]].iloc[-3:].median().to_numpy(), n=len(g)))
    m = len(segs)
    if m == 0:
        return {}
    INF = 1e6
    cost = np.full((m, m), INF)
    for i, a in enumerate(segs):
        for j, b in enumerate(segs):
            if i == j or b["f0"] <= a["f0"] or b["f1"] <= a["f1"]:
                continue
            gap = b["f0"] - a["f1"]
            if gap > max_gap or gap < -max_overlap:
                continue
            if gap <= 0:     # overlap: positions must agree on the shared frames
                A, B = by_seg[a["seg"]], by_seg[b["seg"]]
                common = A.index.intersection(B.index)
                if len(common) == 0:
                    continue
                d = np.linalg.norm(A.loc[common].to_numpy() - B.loc[common].to_numpy(), axis=1).mean()
                if d < overlap_ft:
                    cost[i, j] = d
                continue
            dist = np.linalg.norm(b["p0"] - a["p1"])
            allowed = base_ft + max_speed * gap / 30.0
            if dist <= allowed:
                cost[i, j] = dist / allowed * 10 + gap / 30.0
    big = np.full((m, m), INF)
    np.fill_diagonal(big, new_cost)
    full = np.block([[cost, big], [big, np.zeros((m, m))]])
    r, c = linear_sum_assignment(full)
    nxt = {}
    for i, j in zip(r, c):
        if i < m and j < m and cost[i, j] < INF:
            nxt[segs[i]["seg"]] = segs[j]["seg"]
    prev = {v: k for k, v in nxt.items()}
    ident = {}
    k = 0
    for s_ in sorted(segs, key=lambda s_: s_["f0"]):
        if s_["seg"] in prev:
            continue
        cur = s_["seg"]
        while cur is not None:
            ident[cur] = k
            cur = nxt.get(cur)
        k += 1
    return ident


def resolve_identities(tr: pd.DataFrame, max_per_team=5) -> pd.DataFrame:
    """Return rows with a stable `pid` per player and team in {MIA, LAL}; refs/others dropped."""
    out = []
    pid_base = 0
    for team in ("MIA", "LAL"):
        g = tr[tr.team == team].copy()
        # drop people who live outside the court lines (bench, coaches)
        inside = g.groupby("seg").apply(lambda s: ((s.cy > -0.5) & (s.cy < 50.5) & (s.cx > -1.5)
                                                    & (s.cx < 95.5)).mean(), include_groups=False)
        g = g[g.seg.map(inside) >= 0.5]
        ident = link_segments(g)
        g["pid"] = g.seg.map(ident)
        g = g.sort_values("score", ascending=False).drop_duplicates(["frame", "pid"]).sort_values("frame")
        # remove duplicate boxes of one player (same team, same frame, < 1.5 ft apart)
        g = g.sort_values(["frame", "score"], ascending=[True, False])
        keep = []
        for f, gf in g.groupby("frame"):
            taken = []
            for r in gf.itertuples():
                p = np.array([r.cx, r.cy])
                if any(np.linalg.norm(p - q) < 1.5 for q in taken):
                    continue
                taken.append(p)
                keep.append(r.Index)
        g = g.loc[keep]
        # at most `max_per_team` identities: keep the ones with most frames
        size = g.groupby("pid").size().sort_values(ascending=False)
        if len(size) > max_per_team:
            # merge small identities into a big one when they never overlap in time
            big = list(size.index[:max_per_team])
            for p in size.index[max_per_team:]:
                fr = set(g.frame[g.pid == p])
                best, bd = None, np.inf
                for q in big:
                    if fr & set(g.frame[g.pid == q]):
                        continue
                    a = g[g.pid == p].iloc[0][["cx", "cy"]].to_numpy(float)
                    qq = g[(g.pid == q) & (g.frame < min(fr))]
                    if len(qq) == 0:
                        continue
                    dd = np.linalg.norm(qq.iloc[-1][["cx", "cy"]].to_numpy(float) - a)
                    if dd < bd:
                        best, bd = q, dd
                if best is not None and bd < 15:
                    g.loc[g.pid == p, "pid"] = best
        # hard cap per frame: keep the 5 identities with most support in that frame
        size = g.groupby("pid").size()
        g["support"] = g.pid.map(size)
        g = g.sort_values(["frame", "support"], ascending=[True, False])
        g = g.groupby("frame").head(max_per_team)
        # renumber
        order = g.groupby("pid").frame.min().sort_values().index
        remap = {p: pid_base + i + 1 for i, p in enumerate(order)}
        g["pid"] = g.pid.map(remap)
        pid_base += 10
        out.append(g)
    return pd.concat(out).sort_values(["frame", "pid"]).drop(columns=["support"])
