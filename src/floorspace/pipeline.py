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
    F = teams.features_for_tracks(frames, tr, cam["H"], tex)
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


def stage_identities(clip, model=None, force=False, log=print):
    out = wdir(clip) / "ids.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    cam = court.map_clip(clip, log=log)
    tr = stage_track(clip, log=log)
    F = teams.smooth_features(stage_features(clip, log=log))
    model = model or load_team_model(clip, log=log)
    tr = tr.merge(F, on=["track_id", "frame"])
    tr["team_row"] = teams.label_rows(tr, model)
    xy = np.full((len(tr), 2), np.nan)
    for f, g in tr.groupby("frame"):
        feet = np.c_[(g.x0 + g.x1) / 2, g.y1]
        xy[tr.index.get_indexer(g.index)] = unproject(cam["H"][f], feet)
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
