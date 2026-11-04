"""Rendering: broadcast-style overlay video, minimap, stat panel, end card, debug views.

All vector graphics are drawn with PIL at SS x supersampling and downscaled
(LANCZOS), which gives clean anti-aliased edges. Floor graphics (rings,
spacing polygon) are drawn in court coordinates and warped into the frame with
the per-frame homography, so they sit on the floor in perspective.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from .courtmodel import court_lines, project

SS = 3  # supersampling factor for vector graphics

# palette (RGB)
HEAT_RED = (206, 17, 65)        # Heat red, a touch brighter so it reads on wood
LAKERS_GOLD = (253, 185, 39)
LAKERS_PURPLE = (85, 37, 130)
WHITE = (255, 255, 255)
DEF_GRAY = (232, 234, 238)
PANEL = (11, 15, 20)
CRAMPED = (0xb3, 0x26, 0x1e)
SPREAD = (0x12, 0xa1, 0x50)

TEAM_RGB = {"MIA": HEAT_RED, "LAL": LAKERS_GOLD}


def font(size, weight=600):
    return ImageFont.truetype(str(config.FONTS_DIR / f"Inter-{weight}.ttf"), int(size))


def lerp_color(c0, c1, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def spacing_color(area, lo, hi):
    """Red (cramped) -> green (spread), interpolated in Lab for an even ramp."""
    t = (area - lo) / max(hi - lo, 1e-6)
    a = cv2.cvtColor(np.uint8([[CRAMPED]]), cv2.COLOR_RGB2LAB).astype(float)[0, 0]
    b = cv2.cvtColor(np.uint8([[SPREAD]]), cv2.COLOR_RGB2LAB).astype(float)[0, 0]
    m = a + (b - a) * float(np.clip(t, 0, 1))
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[m]]), cv2.COLOR_LAB2RGB)[0, 0])


# ----------------------------------------------------------------------------
# Minimap
# ----------------------------------------------------------------------------

def _half_lines(side):
    """Court lines of one half as polylines in feet."""
    keep = []
    for kind, name, pts in court_lines():
        if name.startswith(side) or name in ("half_court", "center_circle"):
            if name == "center_circle":
                pts = pts[(pts[:, 0] >= 47) if side == "right" else (pts[:, 0] <= 47)]
            keep.append(pts)
    hx = config.HOOP_FROM_BASELINE if side == "left" else config.COURT_L - config.HOOP_FROM_BASELINE
    a = np.linspace(0, 2 * np.pi, 40)
    keep.append(np.c_[hx + 0.75 * np.cos(a), 25 + 0.75 * np.sin(a)])            # rim
    bx = 4.0 if side == "left" else config.COURT_L - 4.0
    keep.append(np.array([[bx, 22.0], [bx, 28.0]]))                              # backboard
    # dashed inner half of the FT circle
    fx = config.LANE_L if side == "left" else config.COURT_L - config.LANE_L
    sgn = -1 if side == "left" else 1
    for k in range(8):
        t0, t1 = -np.pi / 2 + k * np.pi / 8, -np.pi / 2 + (k + 0.5) * np.pi / 8
        aa = np.linspace(t0, t1, 6)
        keep.append(np.c_[fx + sgn * 6 * np.cos(aa), 25 + 6 * np.sin(aa)])
    return keep


class Minimap:
    """Top-down half court in the same orientation as the broadcast."""

    def __init__(self, side: str, width_px: int, pad_px: int = 18):
        self.side = side
        self.pad = pad_px
        self.x0 = 47.0 if side == "right" else -0.0
        self.x1 = 94.0 if side == "right" else 47.0
        inner_w = width_px - 2 * pad_px
        self.ppf = inner_w / (self.x1 - self.x0 + 2.0)
        self.w = width_px
        self.h = int(round((config.COURT_W + 2.0) * self.ppf + 2 * pad_px))
        self._base = self._draw_base()

    def to_px(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        u = self.pad + (xy[:, 0] - self.x0 + 1.0) * self.ppf
        v = self.pad + (xy[:, 1] + 1.0) * self.ppf
        return np.c_[u, v]

    def _draw_base(self):
        W, H = self.w * SS, self.h * SS
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=16 * SS, fill=PANEL + (184,))
        lw = max(1, int(round(1.6 * SS)))
        for pts in _half_lines(self.side):
            uv = self.to_px(pts) * SS
            d.line([tuple(p) for p in uv], fill=(255, 255, 255, 215), width=lw, joint="curve")
        return im

    def render(self, offense, defense, off_team, poly_color=None, hull=None):
        """offense/defense: lists of (x_ft, y_ft, alpha). Returns RGBA PIL image (w x h)."""
        im = self._base.copy()
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if hull is not None and len(hull) >= 3 and poly_color is not None:
            uv = [tuple(p) for p in self.to_px(hull) * SS]
            d.polygon(uv, fill=poly_color + (92,))
            d.line(uv + [uv[0]], fill=poly_color + (255,), width=int(2.2 * SS), joint="curve")
        im = Image.alpha_composite(im, layer)
        d = ImageDraw.Draw(im)
        r = 1.25 * self.ppf * SS
        for x, y, a in defense:
            u, v = self.to_px([x, y])[0] * SS
            d.ellipse([u - r, v - r, u + r, v + r], fill=DEF_GRAY + (int(235 * a),),
                      outline=(20, 24, 30, int(255 * a)), width=int(1.2 * SS))
        col = TEAM_RGB[off_team]
        for x, y, a in offense:
            u, v = self.to_px([x, y])[0] * SS
            d.ellipse([u - r * 1.1, v - r * 1.1, u + r * 1.1, v + r * 1.1], fill=col + (int(255 * a),),
                      outline=(255, 255, 255, int(255 * a)), width=int(1.6 * SS))
        return im.resize((self.w, self.h), Image.LANCZOS)


# ----------------------------------------------------------------------------
# QA / debug views
# ----------------------------------------------------------------------------

def draw_court_lines(frame, H, color_white=(255, 255, 0), color_purple=(255, 0, 255), width=2):
    """Debug overlay (BGR): cyan = white court lines, magenta = 3-point line."""
    out = frame.copy()
    for kind, name, pts in court_lines():
        uv = project(H, pts)
        ok = np.isfinite(uv).all(1) & (np.abs(uv) < 8000).all(1)
        if ok.sum() < 2:
            continue
        col = color_white if kind == "white" else color_purple
        cv2.polylines(out, [np.round(uv[ok] * 4).astype(np.int32)], False, col, width, cv2.LINE_AA, shift=2)
    return out


def qa_frame(frame, H, rows, side, off_team, label=""):
    """Frame with projected court + foot markers + tags, and the minimap beside it."""
    im = draw_court_lines(frame, H)
    for r in rows:
        u, v = project(H, np.array([[r["x_ft"], r["y_ft"]]]))[0]
        col = TEAM_RGB[r["team"]][::-1] if r["team"] == off_team else (230, 230, 230)
        cv2.circle(im, (int(u), int(v)), 14, col, -1, cv2.LINE_AA)
        cv2.circle(im, (int(u), int(v)), 14, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im, f"{r['pid']}", (int(u) + 16, int(v) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(im, f"{r['pid']}", (int(u) + 16, int(v) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 2, cv2.LINE_AA)
    if label:
        cv2.putText(im, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(im, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
    mm = Minimap(side, 620)
    off = [(r["x_ft"], r["y_ft"], r.get("alpha", 1.0)) for r in rows if r["team"] == off_team]
    de = [(r["x_ft"], r["y_ft"], r.get("alpha", 1.0)) for r in rows if r["team"] != off_team]
    m = mm.render(off, de, off_team)
    # pid tags on the minimap (debug only)
    md = ImageDraw.Draw(m)
    f = font(20, 700)
    for r in rows:
        u, v = mm.to_px([r["x_ft"], r["y_ft"]])[0]
        md.text((u + 12, v - 12), str(r["pid"]), font=f, fill=(255, 255, 255, 255))
    bg = Image.new("RGBA", m.size, (40, 44, 52, 255))
    m = Image.alpha_composite(bg, m).convert("RGB")
    mm_bgr = cv2.cvtColor(np.array(m), cv2.COLOR_RGB2BGR)
    h = im.shape[0]
    canvas = np.full((h, im.shape[1] + mm_bgr.shape[1] + 20, 3), 40, np.uint8)
    canvas[:, :im.shape[1]] = im
    canvas[(h - mm_bgr.shape[0]) // 2:(h - mm_bgr.shape[0]) // 2 + mm_bgr.shape[0], im.shape[1] + 10:im.shape[1] + 10 + mm_bgr.shape[1]] = mm_bgr
    return canvas
