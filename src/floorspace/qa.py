"""Quality checks: court-line debug video, sampled QA frames with minimap, count checks."""
from __future__ import annotations

import subprocess

import cv2
import numpy as np
import pandas as pd

from . import config, court, render
from .pipeline import clip_info, stage_smooth, wdir


def offense_side(sm: pd.DataFrame, off: str) -> str:
    return "right" if sm[sm.team == off].x_ft.median() > 47 else "left"


def court_check_video(clip, out_path=None, log=print):
    """Court lines projected back onto the footage (half resolution, H.264)."""
    cam = court.map_clip(clip, log=log)
    out_path = out_path or (config.OUT_DIR / "debug" / f"{clip}_court_check.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = wdir(clip) / "court_check_raw.mp4"
    cap = cv2.VideoCapture(str(config.MP4_DIR / f"{clip}.mp4"))
    w, h = int(cam["w"]) // 2, int(cam["h"]) // 2
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), config.FPS, (w, h))
    t = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        im = render.draw_court_lines(fr, cam["H"][t], width=3)
        cv2.putText(im, f"{clip}  frame {t:3d}  line fit {cam['score'][t]:.2f}", (30, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
        vw.write(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
        t += 1
    vw.release()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264",
                    "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out_path)], check=True)
    tmp.unlink()
    return out_path


def qa_frames(clip, n=10, log=print):
    """n evenly spaced frames: projected court + markers, and the minimap next to it."""
    info = clip_info(clip)
    cam = court.map_clip(clip, log=log)
    sm = stage_smooth(clip, log=log)
    off = info["offense"]
    side = offense_side(sm, off)
    total = len(cam["H"])
    idx = np.linspace(0, total - 1, n).round().astype(int)
    cap = cv2.VideoCapture(str(config.MP4_DIR / f"{clip}.mp4"))
    outdir = wdir(clip) / "qa"
    outdir.mkdir(exist_ok=True)
    paths = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        rows = sm[(sm.frame == i) & (sm.state != "fading")].to_dict("records")
        q = render.qa_frame(fr, cam["H"][i], rows, side, off, f"{clip} f{i}")
        p = outdir / f"qa_{i:03d}.jpg"
        cv2.imwrite(str(p), q, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p)
    return paths


def counts_report(clip, log=print):
    info = clip_info(clip)
    sm = stage_smooth(clip, log=log)
    n = len(court.map_clip(clip, log=log)["H"])
    tracked = sm[sm.state.isin(["observed", "interpolated", "held"])]
    c = tracked.groupby(["frame", "team"]).size().unstack(fill_value=0).reindex(range(n), fill_value=0)
    for t in ("MIA", "LAL"):
        if t not in c:
            c[t] = 0
    off, de = info["offense"], info["defense"]
    obs = sm[sm.state == "observed"].groupby(["frame", "team"]).size().unstack(fill_value=0)
    obs = obs.reindex(range(n), fill_value=0)
    return {
        "clip": clip, "frames": n,
        "offense_all5_share": round(float((c[off] == 5).mean()), 3),
        "offense_all5_observed_share": round(float((obs.get(off, 0) == 5).mean()), 3),
        "defense_all5_share": round(float((c[de] == 5).mean()), 3),
        "max_offense_per_frame": int(c[off].max()),
        "max_defense_per_frame": int(c[de].max()),
    }
