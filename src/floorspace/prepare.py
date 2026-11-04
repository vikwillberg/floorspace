"""Step 0: probe the raw screen recordings and convert them to clean 30 fps H.264.

* ffprobe each file (codec, size, rotation, frame rate).
* Find the broadcast picture inside the recording: rows/columns that are
  dark in every sampled frame are letterbox bars / phone UI and get cropped.
* Convert to H.264, constant 30 fps, limited-range BT.709, cropped.
* Write a JSON sidecar with the crop and the static overlay mask (scorebug).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class ProbeInfo:
    clip: str
    codec: str
    stored_w: int
    stored_h: int
    rotation: int
    display_w: int
    display_h: int
    avg_fps: float
    nb_frames: int
    duration: float
    size_mb: float


def ffprobe(path: Path) -> ProbeInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,nb_frames:stream_side_data=rotation:"
         "format=duration,size", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    rot = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    num, den = (int(v) for v in s["avg_frame_rate"].split("/"))
    w, h = int(s["width"]), int(s["height"])
    dw, dh = (h, w) if abs(rot) in (90, 270) else (w, h)
    return ProbeInfo(clip=path.stem, codec=s["codec_name"], stored_w=w, stored_h=h,
                     rotation=rot, display_w=dw, display_h=dh,
                     avg_fps=round(num / den, 2), nb_frames=int(s.get("nb_frames", 0)),
                     duration=float(j["format"]["duration"]),
                     size_mb=round(int(j["format"]["size"]) / 1e6, 1))


def _gray_frames(path: Path, w: int, h: int, fps: float = 4.0) -> np.ndarray:
    """Decode (auto-rotated) grayscale frames at a low rate."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w)


def detect_picture_box(path: Path, info: ProbeInfo, thresh: int = 60) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of the broadcast picture.

    A row/column belongs to the picture if it is bright somewhere in at least
    half of the sampled frames. The bars in these recordings carry a faint
    ambient tint (values up to ~50) from the player app, so the threshold sits
    above that. The result is snapped to 16:9 when it is within a few pixels.
    """
    a = _gray_frames(path, info.display_w, info.display_h)
    row_on = (a.max(axis=2) > thresh).mean(axis=0) > 0.5
    col_on = (a.max(axis=1) > thresh).mean(axis=0) > 0.5
    ys, xs = np.flatnonzero(row_on), np.flatnonzero(col_on)
    x0, x1, y0, y1 = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
    w, h = x1 - x0, y1 - y0
    h169 = int(round(w * 9 / 16 / 2) * 2)
    if abs(h - h169) <= 12:          # snap to exact 16:9, centred on the detected box
        cy = (y0 + y1) / 2
        y0, h = int(round(cy - h169 / 2)), h169
    w -= w % 2
    h -= h % 2
    return int(x0), int(y0), int(w), int(h)


def overlay_mask_boxes(w: int, h: int) -> list[list[int]]:
    """Static broadcast graphics to ignore when looking at the floor.

    The scorebug sits bottom-right; extra panels ("SCORING RUN", "FG THIS QTR")
    slide out to its left on the same band. Boxes are (x0, y0, x1, y1) in the
    cropped frame, defined relative to the picture size so they scale.
    """
    scorebug = [int(0.728 * w), int(0.778 * h), int(0.932 * w), int(0.885 * h)]
    extension = [int(0.335 * w), int(0.778 * h), int(0.732 * w), int(0.885 * h)]
    return [scorebug, extension]


def convert(path: Path, out: Path, box: tuple[int, int, int, int], crf: int = 16) -> None:
    x, y, w, h = box
    vf = (f"crop={w}:{h}:{x}:{y},fps={config.FPS},"
          "scale=in_range=pc:out_range=tv,format=yuv420p")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-vf", vf,
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-color_range", "tv", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
           str(out)]
    subprocess.run(cmd, check=True)


def prepare_clip(raw: Path, force: bool = False) -> dict:
    config.MP4_DIR.mkdir(parents=True, exist_ok=True)
    out = config.MP4_DIR / f"{raw.stem}.mp4"
    meta_path = config.MP4_DIR / f"{raw.stem}.json"
    if out.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text())
    info = ffprobe(raw)
    box = detect_picture_box(raw, info)
    convert(raw, out, box)
    conv = ffprobe(out)
    meta = {"raw": asdict(info), "crop_xywh": list(box),
            "mp4": {"w": conv.display_w, "h": conv.display_h, "fps": conv.avg_fps,
                    "nb_frames": conv.nb_frames, "duration": conv.duration, "codec": conv.codec},
            "overlay_mask_boxes": overlay_mask_boxes(box[2], box[3])}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def load_meta(clip: str) -> dict:
    return json.loads((config.MP4_DIR / f"{clip}.json").read_text())
