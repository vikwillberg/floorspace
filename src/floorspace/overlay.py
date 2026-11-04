"""Broadcast-style overlay video.

Layers, back to front:
  1. footage
  2. floor graphics in perspective (spacing polygon, soft rings), drawn at SS x
     and cut out where players stand (segmentation mask), so they sit *under* the players
  3. player tags (screen space, kept clear of faces and the ball)
  4. HUD: live minimap, stat panel (spacing number, N OUT, sparkline), credit
  5. end card on a frozen last frame
"""
from __future__ import annotations

import subprocess

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from . import config, metrics
from .courtmodel import project
from .render import (CRAMPED, DEF_GRAY, PANEL, SPREAD, TEAM_RGB, WHITE, Minimap, font,
                     spacing_color)

SS = 2                    # supersampling for floor graphics
END_CARD_S = 1.5
TEXT_MUTED = (168, 176, 189)
TEXT_DIM = (120, 128, 140)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _circle(cx, cy, r, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def _rounded_panel(w, h, radius, fill=PANEL + (196,), ss=2):
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=radius * ss, fill=fill)
    return im.resize((w, h), Image.LANCZOS)


def _text_w(f, s):
    b = f.getbbox(s)
    return b[2] - b[0]


def _spaced(draw, xy, text, f, fill, tracking=0.08):
    """Letter-spaced caps label."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch) + tracking * f.size
    return x


def _composite(frame_f, rgba, x0, y0):
    """Alpha-composite an RGBA uint8 array onto a float BGR frame at (x0, y0)."""
    h, w = rgba.shape[:2]
    H, W = frame_f.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return
    sub = rgba[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    a = sub[:, :, 3:4] / 255.0
    frame_f[ya:yb, xa:xb] = frame_f[ya:yb, xa:xb] * (1 - a) + sub[:, :, 2::-1] * a


def _backdrop(frame_f, x, y, w, h, radius, unit, darken=0.30):
    """Frosted-glass backdrop: blur + darken the footage under a panel (rounded corners)."""
    H, W = frame_f.shape[:2]
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xb <= xa or yb <= ya:
        return
    sub = frame_f[ya:yb, xa:xb]
    bl = cv2.GaussianBlur(sub, (0, 0), 16 * unit) * (1 - darken)
    m = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], radius=radius * 2, fill=255)
    m = np.array(m.resize((w, h), Image.LANCZOS), np.float32)[ya - y:yb - y, xa - x:xb - x, None] / 255
    frame_f[ya:yb, xa:xb] = sub * (1 - m) + bl * m


# ----------------------------------------------------------------------------
# floor layer
# ----------------------------------------------------------------------------

class FloorLayer:
    def __init__(self, w, h, unit):
        self.w, self.h, self.u = w, h, unit

    def render(self, H, players, hull, hull_rgb, hull_alpha, off_team):
        """players: dicts with x_ft, y_ft, team, alpha, is_off. Returns (rgba crop, x0, y0) or None."""
        shapes = []
        if hull is not None and len(hull) >= 3:
            shapes.append(project(H, hull))
        rings = []
        for p in players:
            r = 2.1 if p["is_off"] else 1.9
            uv = project(H, _circle(p["x_ft"], p["y_ft"], r))
            uv_soft = project(H, _circle(p["x_ft"], p["y_ft"], r * 1.25))
            rings.append((p, uv, uv_soft))
            shapes.append(uv_soft)
        if not shapes:
            return None
        allp = np.vstack(shapes)
        allp = allp[np.isfinite(allp).all(1)]
        m = int(30 * self.u)
        x0 = int(max(np.floor(allp[:, 0].min()) - m, 0))
        y0 = int(max(np.floor(allp[:, 1].min()) - m, 0))
        x1 = int(min(np.ceil(allp[:, 0].max()) + m, self.w))
        y1 = int(min(np.ceil(allp[:, 1].max()) + m, self.h))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        cw, ch = (x1 - x0) * SS, (y1 - y0) * SS

        def loc(uv):
            return [tuple(q) for q in (uv - [x0, y0]) * SS]

        # soft glow under each player (blurred)
        soft = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        ds = ImageDraw.Draw(soft)
        for p, uv, uv_soft in rings:
            col = TEAM_RGB[p["team"]] if p["is_off"] else DEF_GRAY
            ds.polygon(loc(uv_soft), fill=col + (int((120 if p["is_off"] else 80) * p["alpha"]),))
        soft = soft.filter(ImageFilter.GaussianBlur(7 * SS * self.u))

        crisp = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        dc = ImageDraw.Draw(crisp)
        if hull is not None and len(hull) >= 3:
            pts = loc(project(H, hull))
            dc.polygon(pts, fill=hull_rgb + (int(100 * hull_alpha),))
            dc.line(pts + [pts[0]], fill=hull_rgb + (int(250 * hull_alpha),),
                    width=int(round(4.4 * SS * self.u)), joint="curve")
        crisp = Image.alpha_composite(crisp, soft)
        dc = ImageDraw.Draw(crisp)
        for p, uv, _ in rings:
            a = p["alpha"]
            if p["is_off"]:
                col = TEAM_RGB[p["team"]]
                dc.polygon(loc(uv), fill=col + (int(70 * a),))
                dc.line(loc(uv) + [loc(uv)[0]], fill=(255, 255, 255, int(150 * a)),
                        width=int(round(5.2 * SS * self.u)), joint="curve")
                dc.line(loc(uv) + [loc(uv)[0]], fill=col + (int(255 * a),),
                        width=int(round(3.4 * SS * self.u)), joint="curve")
            else:
                dc.polygon(loc(uv), fill=DEF_GRAY + (int(40 * a),))
                dc.line(loc(uv) + [loc(uv)[0]], fill=DEF_GRAY + (int(225 * a),),
                        width=int(round(2.6 * SS * self.u)), joint="curve")
        crisp = crisp.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        return np.array(crisp), x0, y0


# ----------------------------------------------------------------------------
# HUD
# ----------------------------------------------------------------------------

class StatPanel:
    def __init__(self, unit, off_team, lo, hi, series, shot_frame, n_frames):
        self.u = unit
        self.w, self.h = int(470 * unit), int(232 * unit)
        self.off = off_team
        self.lo, self.hi = lo, hi
        self.series = series          # per-frame hull area (nan when not all 5 tracked)
        self.shot_frame = shot_frame
        self.n = n_frames
        self.base = _rounded_panel(self.w, self.h, int(18 * unit), fill=PANEL + (170,))
        self.radius = int(18 * unit)
        self.f_label = font(15 * unit, 600)
        self.f_big = font(66 * unit, 800)
        self.f_unit = font(21 * unit, 600)
        self.f_out = font(46 * unit, 800)
        self.f_outlbl = font(17 * unit, 700)
        self.f_small = font(13 * unit, 600)
        valid = series[np.isfinite(series)]
        self.ymin = min(float(valid.min()) if len(valid) else lo, lo) * 0.92
        self.ymax = max(float(valid.max()) if len(valid) else hi, hi) * 1.04

    def render(self, t, area, n_out, all5, dim=1.0):
        u = self.u
        im = self.base.copy()
        d = ImageDraw.Draw(im)
        pad = int(22 * u)
        team_col = TEAM_RGB[self.off]
        # header
        d.rounded_rectangle([pad, pad + int(3 * u), pad + int(11 * u), pad + int(14 * u)],
                            radius=int(3 * u), fill=team_col + (255,))
        _spaced(d, (pad + int(19 * u), pad), f"{self.off} OFFENSE  ·  FLOOR SPACING", self.f_label,
                TEXT_MUTED + (255,))
        # big number
        y_big = pad + int(20 * u)
        txt = "—" if area is None else f"{int(round(area)):,}"
        num_col = WHITE + (255 if all5 else 150,)
        d.text((pad - int(3 * u), y_big), txt, font=self.f_big, fill=num_col)
        xw = pad + _text_w(self.f_big, txt) + int(10 * u)
        d.text((xw, y_big + int(42 * u)), "sq ft", font=self.f_unit, fill=TEXT_MUTED + (255,))
        if not all5 and area is not None:
            d.text((xw, y_big + int(16 * u)), "4/5 tracked", font=self.f_small, fill=TEXT_DIM + (255,))
        # N OUT block (right)
        xr = self.w - pad
        if n_out is not None:
            s = str(int(n_out))
            wn = _text_w(self.f_out, s)
            wl = _text_w(self.f_outlbl, "OUT")
            xo = xr - (wn + int(8 * u) + wl)
            d.text((xo, y_big + int(10 * u)), s, font=self.f_out, fill=WHITE + (255 if all5 else 150,))
            d.text((xo + wn + int(8 * u), y_big + int(36 * u)), "OUT", font=self.f_outlbl,
                   fill=TEXT_MUTED + (255,))
            # five dots: filled = behind the arc
            r = int(5 * u)
            gap = int(15 * u)
            yd = y_big + int(78 * u)
            x_start = xr - 4 * gap - 2 * r
            for k in range(5):
                cx = x_start + k * gap + r
                if k < n_out:
                    d.ellipse([cx - r, yd - r, cx + r, yd + r], fill=team_col + (255,))
                else:
                    d.ellipse([cx - r, yd - r, cx + r, yd + r], outline=TEXT_DIM + (255,),
                              width=max(1, int(1.5 * u)))
        # colour scale bar with marker
        yb = y_big + int(96 * u)
        bw = self.w - 2 * pad
        bar = np.zeros((int(6 * u), bw, 4), np.uint8)
        for i in range(bw):
            bar[:, i, :3] = spacing_color(self.lo + (self.hi - self.lo) * i / (bw - 1), self.lo, self.hi)
        bar[:, :, 3] = 235
        barim = Image.fromarray(bar, "RGBA")
        mask = Image.new("L", barim.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw - 1, bar.shape[0] - 1], radius=int(3 * u), fill=255)
        im.paste(barim, (pad, yb), mask)
        d.text((pad, yb + int(9 * u)), "CRAMPED", font=self.f_small, fill=TEXT_DIM + (255,))
        d.text((pad + bw - _text_w(self.f_small, "SPREAD"), yb + int(9 * u)), "SPREAD",
               font=self.f_small, fill=TEXT_DIM + (255,))
        if area is not None:
            mx = pad + int(np.clip((area - self.lo) / (self.hi - self.lo), 0, 1) * (bw - 1))
            d.rounded_rectangle([mx - int(3 * u), yb - int(5 * u), mx + int(3 * u), yb + int(11 * u)],
                                radius=int(2 * u), fill=WHITE + (255,), outline=PANEL + (255,),
                                width=max(1, int(1.5 * u)))
        # sparkline
        ys0, ys1 = yb + int(30 * u), self.h - pad + int(4 * u)
        self._sparkline(im, pad, ys0, self.w - pad, ys1, t)
        if dim < 1.0:
            a = np.array(im)
            a[:, :, 3] = (a[:, :, 3] * dim).astype(np.uint8)
            im = Image.fromarray(a, "RGBA")
        return im

    def _sparkline(self, im, x0, y0, x1, y1, t):
        u = self.u
        W, H = (x1 - x0) * SS, (y1 - y0) * SS
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        n = self.n

        def px(f, v):
            return (f / max(n - 1, 1) * (W - 1),
                    (1 - (v - self.ymin) / (self.ymax - self.ymin)) * (H - 1))
        # baseline grid
        d.line([(0, H - 1), (W - 1, H - 1)], fill=(255, 255, 255, 40), width=SS)
        if self.shot_frame is not None:
            sx = self.shot_frame / max(n - 1, 1) * (W - 1)
            d.line([(sx, 0), (sx, H - 1)], fill=(255, 255, 255, 90), width=SS)
            d.text((sx + 4 * SS, 0), "SHOT", font=font(11 * u * SS, 700), fill=(255, 255, 255, 130))
        s = self.series
        # future part faint, past part coloured
        segs, cur = [], []
        for f in range(n):
            if np.isfinite(s[f]):
                cur.append(f)
            elif cur:
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        for seg in segs:
            pts = [px(f, s[f]) for f in seg]
            if len(pts) >= 2:
                d.line(pts, fill=(255, 255, 255, 45), width=int(2 * SS * u), joint="curve")
        for seg in segs:
            past = [f for f in seg if f <= t]
            for a, b in zip(past[:-1], past[1:]):
                col = spacing_color((s[a] + s[b]) / 2, self.lo, self.hi)
                d.line([px(a, s[a]), px(b, s[b])], fill=col + (255,), width=int(3 * SS * u))
        last = [f for f in range(min(t, n - 1) + 1) if np.isfinite(s[f])]
        if last:
            f = last[-1]
            cx, cy = px(f, s[f])
            r = 4.5 * SS * u
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=WHITE + (255,))
        layer = layer.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        im.alpha_composite(layer, (x0, y0))


# ----------------------------------------------------------------------------
# tags
# ----------------------------------------------------------------------------

def tag_image(text, is_off, team, unit):
    f = font(17 * unit, 700)
    tw = _text_w(f, text)
    w, h = int(tw + 16 * unit), int(24 * unit)
    im = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    bg = TEAM_RGB[team] if is_off else (236, 238, 242)
    fg = (255, 255, 255) if (is_off and team == "MIA") else (16, 20, 26)
    d.rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], radius=h, fill=bg + (240,))
    im = im.resize((w, h), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    b = f.getbbox(text)
    d.text(((w - (b[2] - b[0])) / 2 - b[0], (h - (b[3] - b[1])) / 2 - b[1]), text, font=f, fill=fg + (255,))
    return np.array(im)


def place_tag(foot_uv, tag_wh, avoid_boxes, frame_wh, unit):
    """Below the feet by default (on the floor, never over a face); try sides if blocked."""
    u, v = foot_uv
    tw, th = tag_wh
    W, H = frame_wh
    gap = 26 * unit
    cands = [(u - tw / 2, v + gap), (u + gap, v + gap * 0.3), (u - gap - tw, v + gap * 0.3),
             (u - tw / 2, v + gap + th * 1.3)]
    for x, y in cands:
        r = (x, y, x + tw, y + th)
        if r[0] < 0 or r[1] < 0 or r[2] > W or r[3] > H:
            continue
        if not any(r[0] < b[2] and r[2] > b[0] and r[1] < b[3] and r[3] > b[1] for b in avoid_boxes):
            return int(x), int(y)
    return None


# ----------------------------------------------------------------------------
# end card
# ----------------------------------------------------------------------------

def end_card(w, h, unit, info, summ):
    cw, chh = int(900 * unit), int(318 * unit)
    im = _rounded_panel(cw, chh, int(22 * unit), fill=PANEL + (226,))
    d = ImageDraw.Draw(im)
    pad = int(34 * unit)
    team = info["offense"]
    d.rounded_rectangle([pad, pad + int(4 * unit), pad + int(14 * unit), pad + int(18 * unit)],
                        radius=int(4 * unit), fill=TEAM_RGB[team] + (255,))
    res = {"make": "MAKE", "miss": "MISS", "turnover": "TURNOVER"}.get(info["result"], info["result"].upper())
    _spaced(d, (pad + int(24 * unit), pad), f"{team} OFFENSE  ·  CLIP {info['clip'].upper()}  ·  {res}",
            font(19 * unit, 700), WHITE + (255,))
    _spaced(d, (pad, pad + int(30 * unit)), "HEAT VS LAKERS  ·  NOV 2, 2026  ·  FLOOR SPACING SUMMARY",
            font(13 * unit, 600), TEXT_DIM + (255,))

    def stat(x, y, label, value, unit_txt, big=True):
        _spaced(d, (x, y), label, font(13 * unit, 700), TEXT_MUTED + (255,))
        fv = font((54 if big else 34) * unit, 800)
        d.text((x - int(2 * unit), y + int(18 * unit)), value, font=fv, fill=WHITE + (255,))
        if unit_txt:
            d.text((x + _text_w(fv, value) + int(8 * unit), y + int((52 if big else 30) * unit)), unit_txt,
                   font=font(17 * unit, 600), fill=TEXT_MUTED + (255,))

    def fmt(v, nd=0):
        return "—" if v is None or v != v else (f"{v:,.0f}" if nd == 0 else f"{v:.{nd}f}")
    col = (cw - 2 * pad) // 3
    y1 = pad + int(78 * unit)
    stat(pad, y1, "AVG SPACING", fmt(summ["hull_area_sqft_avg"]), "sq ft")
    stat(pad + col, y1, "MIN SPACING", fmt(summ["hull_area_sqft_min"]), "sq ft")
    if summ["ends_in_shot"]:
        stat(pad + 2 * col, y1, "LAST SECOND BEFORE SHOT", fmt(summ["hull_area_sqft_last1s"]), "sq ft")
    else:
        stat(pad + 2 * col, y1, "LAST SECOND BEFORE SHOT", "—", "no shot")
    y2 = y1 + int(112 * unit)
    stat(pad, y2, "AVG OUTSIDE ARC", fmt(summ["n_outside_3_avg"], 1), "of 5", big=False)
    stat(pad + col, y2, "AVG TEAMMATE DISTANCE", fmt(summ["mean_dist_ft_avg"], 1), "ft", big=False)
    stat(pad + 2 * col, y2, "ALL 5 TRACKED", fmt(100 * summ["share_all5_tracked"]), "% of frames", big=False)
    # colour bar of the average
    return im


# ----------------------------------------------------------------------------
# main render
# ----------------------------------------------------------------------------

def render_clip(clip, info, cam, sm, sp, masks, lo, hi, out_path, log=print, max_frames=None, preview=None):
    """Render the overlay MP4. preview=[frames] writes those frames as PNGs to out_path (a dir) instead."""
    cap = cv2.VideoCapture(str(config.MP4_DIR / f"{clip}.mp4"))
    W, H = int(cam["w"]), int(cam["h"])
    unit = W / 1920.0
    n = len(cam["H"])
    off = info["offense"]
    side = sp.attack_side.iloc[0]
    summ = metrics.summarize(sp, info)
    shot = info.get("shot_t_s")
    shot_frame = int(round(float(shot) * config.FPS)) if str(info["ends_in_shot"]).lower() == "true" else None

    # tag numbers: 1..5 per team in order of first appearance
    first = sm[sm.state == "observed"].groupby("pid").frame.min().sort_values()
    tags = {}
    for team in ("MIA", "LAL"):
        k = 1
        for pid in first.index:
            if sm[sm.pid == pid].team.iloc[0] == team:
                tags[pid] = str(k)
                k += 1

    series = sp.hull_area_sqft.to_numpy(float)
    floor = FloorLayer(W, H, unit)
    panel = StatPanel(unit, off, lo, hi, series, shot_frame, n)
    mm = Minimap(side, int(330 * unit), pad_px=int(16 * unit))
    pad = int(36 * unit)
    mm_xy = (pad, pad) if side == "right" else (W - pad - mm.w, pad)
    panel_xy = (pad, H - pad - panel.h) if side == "right" else (int(W * 0.455), H - pad - panel.h)
    f_credit = font(15 * unit, 600)
    tag_cache = {}
    from .prepare import load_meta
    gfx_boxes = load_meta(clip)["overlay_mask_boxes"][:1]          # the scorebug itself
    gfx_mask = np.zeros((H, W), np.float32)
    for gx0, gy0, gx1, gy1 in gfx_boxes:
        gfx_mask[gy0:gy1, gx0:gx1] = 1.0
    d = np.load(config.WORK_DIR / clip / "detections.npz")
    ball = {int(f): b for f, b, c, k in zip(d["frame"], d["xyxy"], d["conf"], d["cls"]) if k == 32 and c > 0.2}

    if preview is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        ff = None
    else:
        ff = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(config.FPS), "-i", "-", "-c:v", "libx264",
                           "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
                           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                           "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    preview = set(preview) if preview is not None else None
    hull_fade = 0.0
    last_area = None
    last_out = None
    last_frame = None
    total = n if max_frames is None else min(n, max_frames)
    for t in range(total):
        ok, fr = cap.read()
        if not ok:
            break
        row = sp.iloc[t]
        if row.all5_tracked:
            last_area, last_out = float(row.hull_area_sqft), int(row.n_outside_3)
        if preview is not None and t not in preview and t != total - 1:
            continue
        Hm = cam["H"][t]
        rows = sm[(sm.frame == t) & (sm.alpha > 0.02)]
        players = [dict(x_ft=r.x_ft, y_ft=r.y_ft, team=r.team, alpha=float(r.alpha), pid=int(r.pid),
                        is_off=(r.team == off)) for r in rows.itertuples()]
        all5 = bool(row.all5_tracked)
        hull_fade = min(1.0, hull_fade + 0.15) if all5 else max(0.35, hull_fade - 0.08)
        offp = np.array([[p["x_ft"], p["y_ft"]] for p in players if p["is_off"] and p["alpha"] > 0.5])
        hull = metrics.hull_polygon(offp) if len(offp) >= 3 else None
        if preview is not None:
            hull_fade = 1.0 if all5 else 0.35
        area_disp = last_area
        hull_rgb = spacing_color(area_disp if area_disp is not None else lo, lo, hi)

        out = fr.astype(np.float32)
        layer = floor.render(Hm, players, hull, hull_rgb, hull_fade if hull is not None else 0, off)
        if layer is not None:
            rgba, x0, y0 = layer
            if masks is not None or gfx_mask is not None:
                mk = np.zeros(rgba.shape[:2], np.float32)
                if masks is not None:
                    mk = masks[t, y0:y0 + rgba.shape[0], x0:x0 + rgba.shape[1]].astype(np.float32)
                mk = np.maximum(mk, gfx_mask[y0:y0 + rgba.shape[0], x0:x0 + rgba.shape[1]])
                mk = cv2.GaussianBlur(mk, (0, 0), 1.2 * unit)
                rgba = rgba.copy()
                rgba[:, :, 3] = (rgba[:, :, 3] * (1 - np.clip(mk * 1.15, 0, 1))).astype(np.uint8)
            _composite(out, rgba, x0, y0)

        # tags: avoid faces (top 22% of every on-court person box) and the ball
        avoid = [tuple(b) for b in gfx_boxes]
        tb = sm_boxes(t)
        for b in tb:
            x0b, y0b, x1b, y1b = b
            avoid.append((x0b, y0b, x1b, y0b + 0.22 * (y1b - y0b)))
        if t in ball:
            bx = ball[t]
            m_ = 10 * unit
            avoid.append((bx[0] - m_, bx[1] - m_, bx[2] + m_, bx[3] + m_))
        for p in players:
            key = (tags.get(p["pid"], "?"), p["is_off"], p["team"])
            if key not in tag_cache:
                tag_cache[key] = tag_image(key[0], p["is_off"], p["team"], unit)
            ti = tag_cache[key]
            uv = project(Hm, np.array([[p["x_ft"], p["y_ft"]]]))[0]
            pos = place_tag(uv, (ti.shape[1], ti.shape[0]), avoid, (W, H), unit)
            if pos is None:
                continue
            ti2 = ti.copy()
            ti2[:, :, 3] = (ti2[:, :, 3] * p["alpha"]).astype(np.uint8)
            _composite(out, ti2, *pos)
            avoid.append((pos[0], pos[1], pos[0] + ti.shape[1], pos[1] + ti.shape[0]))

        # HUD
        offense = [(p["x_ft"], p["y_ft"], p["alpha"]) for p in players if p["is_off"]]
        defense = [(p["x_ft"], p["y_ft"], p["alpha"]) for p in players if not p["is_off"]]
        mmi = mm.render(offense, defense, off, poly_color=hull_rgb, hull=hull)
        _backdrop(out, *mm_xy, mm.w, mm.h, int(16 * unit), unit)
        _composite(out, np.array(mmi), *mm_xy)
        pim = panel.render(t, area_disp, last_out, all5)
        _backdrop(out, *panel_xy, panel.w, panel.h, panel.radius, unit)
        _composite(out, np.array(pim), *panel_xy)
        cred = Image.new("RGBA", (int(220 * unit), int(28 * unit)), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cred)
        txt = "Viktor Berg"
        cd.text((cred.width - _text_w(f_credit, txt) - 2, 2), txt, font=f_credit, fill=(255, 255, 255, 190))
        _composite(out, np.array(cred), W - pad - cred.width, H - int(10 * unit) - cred.height)

        frame_out = out.clip(0, 255).astype(np.uint8)
        if ff is None:
            if t in preview:
                cv2.imwrite(str(out_path / f"{clip}_{t:03d}.png"), frame_out)
        else:
            ff.stdin.write(frame_out.tobytes())
        last_frame = frame_out
        if t % 30 == 0:
            log(f"  {clip} render frame {t}/{total}")

    # end card: freeze the last frame, dim it, card fades/slides in
    card = end_card(W, H, unit, info, summ)
    if ff is None:
        f2 = last_frame.astype(np.float32) * 0.45 + np.array(PANEL[::-1], np.float32) * 0.55
        _backdrop(f2, (W - card.width) // 2, (H - card.height) // 2, card.width, card.height, int(22 * unit), unit)
        _composite(f2, np.array(card), (W - card.width) // 2, (H - card.height) // 2)
        cv2.imwrite(str(out_path / f"{clip}_endcard.png"), f2.clip(0, 255).astype(np.uint8))
        return out_path
    nf = int(END_CARD_S * config.FPS)
    base = last_frame.astype(np.float32)
    for k in range(nf):
        a = min(1.0, (k + 1) / 9)
        e = 1 - (1 - a) ** 3
        f2 = base * (1 - 0.55 * e) + np.array(PANEL[::-1], np.float32) * 0.55 * e
        ci = np.array(card).copy()
        ci[:, :, 3] = (ci[:, :, 3] * e).astype(np.uint8)
        y = int((H - card.height) / 2 + (1 - e) * 30 * unit)
        if e > 0.98:
            _backdrop(f2, (W - card.width) // 2, y, card.width, card.height, int(22 * unit), unit)
        _composite(f2, ci, (W - card.width) // 2, y)
        ff.stdin.write(f2.clip(0, 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    return out_path


_SM_BOXES = {}


def sm_boxes(t):
    return _SM_BOXES.get(t, [])


def set_person_boxes(per_frame_boxes):
    _SM_BOXES.clear()
    for t, b in enumerate(per_frame_boxes):
        _SM_BOXES[t] = [tuple(x) for x in b]
