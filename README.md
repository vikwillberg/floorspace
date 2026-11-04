# floorspace

Work in progress: NBA floor-spacing tracker for broadcast clips (Heat vs Lakers, Nov 2 2026).

Status: clip preparation, player tracking and court mapping run on p01. Overlay rendering and full results are next.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m floorspace prepare            # raw .mov -> clips/mp4/*.mp4 (30 fps H.264, cropped)
python -m floorspace run clips/raw/p01.mov
python -m floorspace qa p01             # court-check video + QA frames
```
