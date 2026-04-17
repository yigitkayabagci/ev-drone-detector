# Frame-based Detector Bridge (FRED-style)

Utilities for running a **frame-based** detector (e.g. a model trained on the
[FRED](https://github.com/miccunifi/FRED) event-frame dataset) against the
event-stream EV-UAV `.npz` data used by the rest of this repo.

## Why this exists

- EV-UAV provides raw asynchronous events (`ev_loc`, `evs_norm` per `.npz`).
- FRED-trained detectors consume **2D event frames** (one image per 33.3 ms
  bin at 30 FPS, or a single collapsed frame per sample).

`events_to_frames.py` is the bridge: it bins the events and writes standard
image tensors a frame-based model can load.

## 1. Convert EV-UAV events to frames

```bash
# One collapsed PNG per .npz (quick start)
uv run python frame_detector/events_to_frames.py \
    --input data/test \
    --output frames/test \
    --mode single --format png --channels polarity_rgb

# 30-FPS windowed, numpy tensors for direct network input
uv run python frame_detector/events_to_frames.py \
    --input data/test \
    --output frames/test \
    --mode windowed --window_ms 33.3 \
    --format npy --channels counts_2ch
```

### Output layouts

- `--mode single`: one file per `.npz` directly under `--output`.
  ```
  frames/test/seq001_0000.png
  frames/test/seq001_0001.png
  ```
- `--mode windowed`: one sub-folder per `.npz`, N frames inside.
  ```
  frames/test/seq001_0000/f0000.png
  frames/test/seq001_0000/f0001.png
  ...
  ```

### Channel layouts

| `--channels`     | Shape           | dtype   | Use case |
| ---------------- | --------------- | ------- | -------- |
| `polarity_rgb`   | `(H, W, 3)`     | uint8   | Drop-in RGB network input, PNG visualizable |
| `counts_2ch`     | `(2, H, W)`     | float32 | Event-native CNN detectors (pos / neg count channels) |

Default resolution is DAVIS346 (346×260). Override with `--resolution W H`.

## 2. Running your FRED-trained model — what we need from you

This folder does **not** yet include the model side. FRED itself is a dataset
(no baseline weights), so I cannot infer your friend's architecture. To
finish the bridge please share:

1. **Model source** — the Python class / package and how to construct it
   (e.g. `from fred_det.models import TinyDet; model = TinyDet(...)`), or a
   checkpoint that includes the architecture.
2. **Checkpoint path** — `.pt` / `.pth` file.
3. **Input tensor spec** — shape, dtype, normalization:
   - Does it want `(3, H, W)` RGB-like or `(2, H, W)` event counts?
   - Expected H, W (and whether it handles 346×260 natively or needs resize)?
   - Any mean/std normalization?
4. **Output format** — bounding boxes in `[x1, y1, x2, y2]`, `[cx, cy, w, h]`,
   YOLO-normalized, or raw logits with a decoder? One detection per frame or
   multi-drone?

Once I have those, I'll add `frame_detector/infer.py` that loads your
checkpoint, feeds frames produced by `events_to_frames.py`, and writes
the same JSON + MP4 outputs as `scripts/detect.py` so the two detectors
are directly comparable on EV-UAV.
