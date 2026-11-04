"""Tidy tables for SQL / Power BI and the README charts.

data/tracking.csv       one row per player per frame (court feet)
data/spacing.csv        one row per frame (offense metrics)
data/clips_summary.csv  one row per clip
docs/spacing_by_clip.png, docs/spacing_over_time.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, metrics
from .pipeline import clip_info, clip_table, stage_metrics, stage_smooth

CHART_RED = "#CE1141"      # Heat
CHART_GOLD = "#C99400"     # Lakers gold, darkened for contrast on a light surface
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
TEAM_CHART = {"MIA": CHART_RED, "LAL": CHART_GOLD}


def tracking_table(clip) -> pd.DataFrame:
    info = clip_info(clip)
    sm = stage_smooth(clip)
    sm = sm[sm.state.isin(metrics.TRACKED_STATES)].copy()
    out = pd.DataFrame({
        "clip": clip, "frame": sm.frame, "t_sec": (sm.frame / config.FPS).round(3),
        "track_id": sm.pid.astype(int), "team": sm.team,
        "side": np.where(sm.team == info["offense"], "offense", "defense"),
        "x_ft": sm.x_ft.round(2), "y_ft": sm.y_ft.round(2),
        "visible": (sm.state == "observed").astype(int)})
    return out.sort_values(["frame", "track_id"])


def export_all(clips=None, log=print):
    clips = clips or list(clip_table()["clip"])
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tr = pd.concat([tracking_table(c) for c in clips])
    tr.to_csv(config.DATA_DIR / "tracking.csv", index=False)
    sp = pd.concat([stage_metrics(c) for c in clips])
    sp.to_csv(config.DATA_DIR / "spacing.csv", index=False)
    summ = pd.DataFrame([metrics.summarize(stage_metrics(c), clip_info(c)) for c in clips])
    summ.to_csv(config.DATA_DIR / "clips_summary.csv", index=False)
    log(f"  data/: tracking.csv ({len(tr):,} rows), spacing.csv ({len(sp):,}), clips_summary.csv ({len(summ)})")
    return tr, sp, summ


def _style():
    import matplotlib as mpl
    from matplotlib import font_manager
    for w in (400, 500, 600, 700):
        font_manager.fontManager.addfont(str(config.FONTS_DIR / f"Inter-{w}.ttf"))
    mpl.rcParams.update({
        "font.family": "Inter", "font.size": 11, "text.color": TEXT, "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True})


def chart_by_clip(summ: pd.DataFrame, path):
    """Average spacing per clip (bar) + last second before the shot (dot), grouped by offense."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _style()
    s = summ.copy()
    s["order"] = s.offense.map({"MIA": 0, "LAL": 1})
    s = s.sort_values(["order", "hull_area_sqft_avg"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(s) + 1.9), dpi=200)
    y = np.arange(len(s))[::-1]
    for yi, r in zip(y, s.itertuples()):
        col = TEAM_CHART[r.offense]
        ax.barh(yi, r.hull_area_sqft_avg, height=0.58, color=col, edgecolor=SURFACE, linewidth=2)
        lab = f"{r.hull_area_sqft_avg:,.0f}"
        ax.text(r.hull_area_sqft_avg + 12, yi, lab, va="center", ha="left", fontsize=10, color=TEXT,
                fontweight=600)
        if r.hull_area_sqft_last1s == r.hull_area_sqft_last1s:
            ax.plot(r.hull_area_sqft_last1s, yi, "o", ms=8, mfc=SURFACE, mec=TEXT, mew=1.6, zorder=5)
    names = [f"{r.clip}  ·  {r.result}" for r in s.itertuples()]
    ax.set_yticks(y, names, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Offensive spacing: convex hull of the 5 players (sq ft)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(s.hull_area_sqft_avg.max(), s.hull_area_sqft_last1s.max(skipna=True)) * 1.15)
    team_avg = s.groupby("offense").hull_area_sqft_avg.mean()
    ttl = "Floor spacing per clip  ·  Heat vs Lakers, Nov 2 2026"
    fig.text(0.012, 0.985, ttl, ha="left", va="top", fontsize=14, fontweight=700)
    sub = "  ·  ".join(f"{t} offense avg {team_avg[t]:,.0f} sq ft ({(s.offense == t).sum()} clips)"
                        for t in ("MIA", "LAL") if t in team_avg)
    fig.text(0.012, 0.935, sub, ha="left", va="top", fontsize=10, color=TEXT2)
    handles = [Patch(color=CHART_RED, label="MIA offense (clip average)"),
               Patch(color=CHART_GOLD, label="LAL offense (clip average)"),
               Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=TEXT, mew=1.6,
                      label="last second before the shot")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def chart_over_time(sp: pd.DataFrame, summ: pd.DataFrame, path):
    """Small multiples: spacing per frame in the analysis window, shot release marked."""
    import matplotlib.pyplot as plt
    _style()
    clips = list(summ.sort_values(["offense", "clip"], ascending=[False, True]).clip)
    n = len(clips)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.3 * rows + 0.8), dpi=200, sharey=True)
    axes = np.atleast_2d(axes)
    ymax = sp[sp.in_window].hull_area_sqft.max() * 1.1
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        c = clips[k]
        info = clip_info(c)
        g = sp[(sp.clip == c) & sp.in_window]
        col = TEAM_CHART[info["offense"]]
        ax.plot(g.t_sec, g.hull_area_sqft, color=col, lw=2, solid_capstyle="round")
        if str(info["ends_in_shot"]).lower() == "true":
            ax.axvline(float(info["shot_t_s"]), color=TEXT2, lw=1, ls=(0, (2, 2)))
        ax.set_title(f"{c} · {info['offense']} · {info['result']}", fontsize=10, loc="left", fontweight=600)
        ax.set_ylim(0, ymax)
        ax.tick_params(labelsize=8, length=0)
        ax.grid(axis="x", visible=False)
        if k % cols == 0:
            ax.set_ylabel("sq ft", fontsize=9)
        if k >= (rows - 1) * cols:
            ax.set_xlabel("seconds", fontsize=9)
    fig.text(0.01, 0.99, "Spacing through each possession (dashed line = shot release; gaps = fewer than 5 tracked)",
             ha="left", va="top", fontsize=12, fontweight=700)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def charts(sp, summ):
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart_by_clip(summ, config.DOCS_DIR / "spacing_by_clip.png")
    chart_over_time(sp, summ, config.DOCS_DIR / "spacing_over_time.png")
