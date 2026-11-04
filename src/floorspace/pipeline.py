"""Per-clip pipeline with cached stages under work/<clip>/.

  prepare -> detect -> camera -> track -> features -> identities -> smooth -> metrics -> render
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, court, detect_track, smoothing, teams
from .courtmodel import unproject
from .prepare import load_meta, prepare_clip

TEAM_MODEL_JSON = config.DATA_DIR / "team_model.json"


def clip_table() -> pd.DataFrame:
    return pd.read_csv(config.CLIPS_CSV)


def clip_info(clip: str) -> dict:
    t = clip_table()
    return t[t["clip"] == clip].iloc[0].to_dict()


def wdir(clip):
    d = config.WORK_DIR / clip
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_track(clip, force=False, log=print):
    out = wdir(clip) / "tracks_raw.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    cam = court.map_clip(clip, log=log)
    tr = detect_track.track_clip(clip, cam["H"], cam["P"], load_meta(clip)["overlay_mask_boxes"])
    tr.to_csv(out, index=False)
    log(f"  {clip}: {tr.track_id.nunique()} ByteTrack tracks, {len(tr)} rows")
    return tr


def stage_features(clip, force=False, log=print, frames=None):
    out = wdir(clip) / "features.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    cam = court.map_clip(clip, log=log)
    tr = stage_track(clip, log=log)
    frames = frames if frames is not None else court.read_frames(clip)
    boxes = court.person_boxes(clip, len(frames))
    tex = teams.build_court_texture(frames, cam["H"], boxes)
    import cv2
    cv2.imwrite(str(wdir(clip) / "court_texture.jpg"), tex[0])
    gfx_boxes = load_meta(clip)["overlay_mask_boxes"][:1]
    tr = tr.reset_index(drop=True)
    _, vfull = foot_positions(tr, cam, gfx_boxes, return_v=True)
    tr_ext = tr.assign(y1=np.maximum(tr.y1.to_numpy(), vfull))   # boxes cut off by the scorebug, extended
    F = teams.features_for_tracks(frames, tr_ext, cam["H"], tex, gfx_boxes=gfx_boxes)
    F.to_csv(out, index=False)
    return F


def fit_team_model(clips, log=print):
    F = pd.concat([teams.smooth_features(stage_features(c, log=log)).assign(clip=c) for c in clips])
    model = teams.fit_clusters(F)
    model["clips"] = list(clips)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEAM_MODEL_JSON.write_text(json.dumps(model, indent=2))
    log(f"  team colour model fitted on {model['n_rows']} detections from {len(clips)} clip(s)")
    return model


def load_team_model(clip, log=print):
    if TEAM_MODEL_JSON.exists():
        return json.loads(TEAM_MODEL_JSON.read_text())
    return fit_team_model([clip], log=log)


PERSON_HEIGHT_FT = 6.6   # head-top height used when the feet are hidden


def foot_positions(tr: pd.DataFrame, cam, gfx_boxes, return_v=False):
    """Court position of each box's feet (bottom-centre of the box).

    When the bottom of a box is cut off by a broadcast graphic (the scorebug), the
    feet are estimated from the top of the box instead: find the floor point
    whose projected 6.6 ft person top lands on the box top.
    """
    xy = np.full((len(tr), 2), np.nan)
    vfull = np.full(len(tr), np.nan)
    for f, g in tr.groupby("frame"):
        H, P = cam["H"][f], cam["P"][f]
        u = ((g.x0 + g.x1) / 2).to_numpy()
        vb = g.y1.to_numpy().astype(float)
        vt = g.y0.to_numpy().astype(float)
        cut = np.zeros(len(g), bool)
        for gx0, gy0, gx1, gy1 in gfx_boxes:
            cut |= (u > gx0 - 10) & (u < gx1 + 10) & (vb > gy0 - 14) & (vb < gy1)
        for k in np.flatnonzero(cut):
            v = vb[k]
            for _ in range(6):
                X = unproject(H, np.array([[u[k], v]]))[0]
                top = P @ np.array([X[0], X[1], -PERSON_HEIGHT_FT, 1.0])
                v_top = top[1] / top[2]
                v = v + (vt[k] - v_top)
            vb[k] = max(v, vb[k])
        idx = tr.index.get_indexer(g.index)
        xy[idx] = unproject(H, np.c_[u, vb])
        vfull[idx] = vb
    return (xy, vfull) if return_v else xy


def stage_identities(clip, model=None, force=False, log=print):
    out = wdir(clip) / "ids.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    cam = court.map_clip(clip, log=log)
    tr = stage_track(clip, log=log)
    F = teams.smooth_features(stage_features(clip, log=log))
    model = model or load_team_model(clip, log=log)
    tr = tr.merge(F, on=["track_id", "frame"]).reset_index(drop=True)
    tr["team_row"] = teams.label_rows(tr, model)
    xy = foot_positions(tr, cam, load_meta(clip)["overlay_mask_boxes"][:1])
    tr["cx"], tr["cy"] = xy[:, 0], xy[:, 1]
    seg = teams.split_segments(tr)
    ids = teams.resolve_identities(seg)
    keep = ["frame", "pid", "team", "track_id", "seg", "x0", "y0", "x1", "y1", "score", "cx", "cy"]
    ids[keep].to_csv(out, index=False)
    log(f"  {clip}: identities {ids.groupby('team').pid.nunique().to_dict()}")
    return ids[keep]


def stage_smooth(clip, force=False, log=print):
    out = wdir(clip) / "smooth.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    ids = stage_identities(clip, log=log)
    n = detect_track.n_frames(clip)
    sm = smoothing.cap_per_team(smoothing.smooth_tracks(ids, n))
    sm.to_csv(out, index=False)
    return sm


def prepare_all(log=print):
    for raw in sorted(config.RAW_DIR.glob("*.mov")):
        prepare_clip(raw)


# ----------------------------------------------------------------------------
# metrics + render
# ----------------------------------------------------------------------------

THRESH_JSON = config.DATA_DIR / "spacing_thresholds.json"


def stage_metrics(clip, log=print):
    from . import metrics
    info = clip_info(clip)
    sm = stage_smooth(clip, log=log)
    n = detect_track.n_frames(clip)
    sp = metrics.per_frame(clip, sm, info, n)
    sp.to_csv(wdir(clip) / "spacing.csv", index=False)
    return sp


def fit_thresholds(clips, log=print):
    """Colour-scale ends from the data: 10th / 90th percentile of in-window hull area over `clips`."""
    vals = []
    for c in clips:
        sp = stage_metrics(c, log=log)
        vals.append(sp[sp.in_window & sp.all5_tracked].hull_area_sqft)
    v = pd.concat(vals)
    th = {"cramped_sqft": round(float(v.quantile(0.10)), 0), "spread_sqft": round(float(v.quantile(0.90)), 0),
          "rule": "10th / 90th percentile of per-frame hull area (in-window, all 5 tracked)",
          "clips": list(clips), "n_frames": int(len(v))}
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    THRESH_JSON.write_text(json.dumps(th, indent=2))
    log(f"  colour scale: red at {th['cramped_sqft']:.0f} sq ft, green at {th['spread_sqft']:.0f} sq ft "
        f"({th['n_frames']} frames, {len(clips)} clip(s))")
    return th


def stage_render(clip, log=print, preview=None):
    from . import overlay
    info = clip_info(clip)
    cam = court.map_clip(clip, log=log)
    sm = stage_smooth(clip, log=log)
    sp = stage_metrics(clip, log=log)
    th = json.loads(THRESH_JSON.read_text()) if THRESH_JSON.exists() else fit_thresholds([clip], log=log)
    masks = detect_track.player_masks(clip, cam["H"], cam["P"], log=log)
    ids = stage_identities(clip, log=log)
    n = len(cam["H"])
    overlay.set_person_boxes([ids[ids.frame == t][["x0", "y0", "x1", "y1"]].to_numpy() for t in range(n)])
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUT_DIR / f"{clip}_floorspace.mp4" if preview is None else wdir(clip) / "preview"
    overlay.render_clip(clip, info, cam, sm, sp, masks, th["cramped_sqft"], th["spread_sqft"], out,
                        log=log, preview=preview)
    return out


def make_gif(mp4, gif, width=720, fps=12, start=0.0, dur=8.2):
    """Looping, palette-optimised README GIF."""
    import subprocess
    # light denoise + no dithering keeps the noisy crowd from blowing up the file (target <= 10 MB)
    vf = f"fps={fps},scale={width}:-2:flags=lanczos,hqdn3d=5:5:8:8"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start), "-t", str(dur), "-i", str(mp4),
                    "-vf", f"{vf},split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];"
                           f"[b][p]paletteuse=dither=none:diff_mode=rectangle",
                    "-loop", "0", str(gif)], check=True)
    return gif
