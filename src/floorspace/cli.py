"""Command line entry point: python -m floorspace <command>."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def _clip_names(arg: str | None) -> list[str]:
    if arg:
        return [Path(arg).stem]
    return sorted(p.stem for p in config.RAW_DIR.glob("*.mov"))


def cmd_prepare(args):
    from .prepare import prepare_clip
    for c in _clip_names(args.clip):
        meta = prepare_clip(config.RAW_DIR / f"{c}.mov", force=args.force)
        r, m = meta["raw"], meta["mp4"]
        print(f"{c}: {r['codec']} {r['stored_w']}x{r['stored_h']} rot={r['rotation']} "
              f"~{r['avg_fps']}fps VFR -> crop {meta['crop_xywh']} -> {m['codec']} "
              f"{m['w']}x{m['h']} {m['fps']}fps {m['nb_frames']} frames")


def cmd_calibrate(args):
    from . import court
    kfs = [(k.split(":")[0], int(k.split(":")[1])) for k in args.keyframes]
    court.calibrate(kfs)


STAGES = ["camera", "track", "features", "identities", "smooth"]


def _forces(force_from):
    return {s: (force_from is not None and STAGES.index(s) >= STAGES.index(force_from)) for s in STAGES}


def _run_front(clip, force):
    """Stages that do not depend on other clips: prepare, detect, camera, track, colour features."""
    from . import court, detect_track, pipeline
    from .prepare import prepare_clip
    print(f"== {clip}")
    prepare_clip(config.RAW_DIR / f"{clip}.mov")
    detect_track.detect(clip)
    court.map_clip(clip, force=force["camera"])
    pipeline.stage_track(clip, force=force["track"])
    pipeline.stage_features(clip, force=force["features"])


def _run_back(clip, force, model=None):
    from . import pipeline, qa as qamod
    pipeline.stage_identities(clip, model=model, force=force["identities"])
    pipeline.stage_smooth(clip, force=force["smooth"])
    print("  ", json.dumps(qamod.counts_report(clip)))


def cmd_run(args):
    force = _forces(args.force_from)
    for c in _clip_names(args.clip):
        _run_front(c, force)
        _run_back(c, force)


def cmd_run_all(args):
    from . import pipeline
    force = _forces(args.force_from)
    clips = _clip_names(None)
    for c in clips:
        _run_front(c, force)
    # one team-colour model for the whole game, fitted on every clip
    model = pipeline.fit_team_model(clips)
    force["identities"] = force["smooth"] = True
    for c in clips:
        _run_back(c, force, model)


def cmd_qa(args):
    from . import qa
    for c in _clip_names(args.clip):
        print(qa.court_check_video(c))
        for p in qa.qa_frames(c, n=args.n):
            print(p)
        print(json.dumps(qa.counts_report(c)))


def cmd_render(args):
    from . import pipeline
    for c in _clip_names(args.clip):
        out = pipeline.stage_render(c)
        print(out)
        if args.gif:
            config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
            print(pipeline.make_gif(out, config.DOCS_DIR / f"{c}_floorspace.gif"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floorspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="probe + convert raw clips to 30 fps H.264 MP4")
    p.add_argument("clip", nargs="?")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("calibrate", help="estimate the broadcast camera position from keyframes")
    p.add_argument("keyframes", nargs="+", help="clip:frame, e.g. p01:0 p05:0")
    p.set_defaults(func=cmd_calibrate)
    for name, fn in (("run", cmd_run), ("run-all", cmd_run_all)):
        p = sub.add_parser(name, help="run the pipeline" + (" on one clip" if name == "run" else " on all clips"))
        if name == "run":
            p.add_argument("clip")
        p.add_argument("--force-from", choices=["camera", "track", "features", "identities", "smooth"])
        p.set_defaults(func=fn)
    p = sub.add_parser("qa", help="court-check video + sampled QA frames + count report")
    p.add_argument("clip", nargs="?")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(func=cmd_qa)
    p = sub.add_parser("render", help="render the overlay video (outputs/<clip>_floorspace.mp4)")
    p.add_argument("clip", nargs="?")
    p.add_argument("--gif", action="store_true", help="also write docs/<clip>_floorspace.gif")
    p.set_defaults(func=cmd_render)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
