# Retromod Camera

A Raspberry Pi Zero 2 W built into an old film camera body, with film simulations
in the spirit of modern Fujifilm and Leica digital cameras. Six sims are built in,
selected by swiping the small rear touchscreen. There is no live preview on the
device — photos are processed after capture, and the raw frame is always kept so
shots can be re-developed later with different settings.

## Phases

1. **macOS development (this code)** — the simulation engine plus a live preview
   window using the Mac webcam or a folder of sample images, and a mock of the
   240x240 rear display. This will grow into a profile *editor*: tune a sim's
   look interactively and save it as JSON.
2. **Pi hardware** — Picamera2 capture on a shutter button, post-capture
   processing, ST7789 touchscreen showing the film-box UI, swipe to change sims.
3. **Polish** — battery gauge, re-develop mode, photo transfer.

## Running (macOS)

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Controls: `SPACE` capture, `1`-`6` select sim, `,` / `.` cycle sims, `Q` quit.
Set `INPUT_SOURCE = "folder"` in `config.py` to test against images in `samples/`
instead of the webcam.

## Film profiles

Each sim is a JSON file in `profiles/` — the filename stem is the profile key,
and `order` sets the swipe sequence. Both the Mac app and (later) the Pi load
the same files, so tuning a look on the Mac and copying the file to the camera
is the whole deployment story.

| Field | Meaning |
|---|---|
| `name`, `subtitle` | Shown on the rear display |
| `box_color` | RGB colour of the film-box UI |
| `color_matrix` | 3x3 RGB transform |
| `tone_curve` | `[x, y]` control points (0-1), smoothly interpolated; omit for linear |
| `contrast`, `saturation`, `brightness` | Global adjustments |
| `shadow_lift` | Lifts crushed blacks |
| `monochrome` | Black & white sim |
| `grain_strength` | 0 = none, ~0.3 = heavy |

## Tests

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```
