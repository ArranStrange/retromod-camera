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

Controls: `SPACE` capture, `1`-`6` select sim, `,` / `.` cycle sims, `B` toggle
an original/sim split view, `Q` quit. Set `INPUT_SOURCE = "folder"` in
`config.py` to test against images in `samples/` instead of the webcam.

### Tuning studio

Press `E` to edit the current sim live: `P` cycles parameter pages (Tone,
Colour, Grade, HSL Hue/Sat/Lum, Effects), `[` / `]` select a parameter,
`-` / `=` adjust it, `S` write it back to the profile's JSON file, `E` again
to exit (unsaved tweaks are discarded). `warmth` (red-blue) and `tint`
(green-magenta) are one-value colour shifts baked into the colour matrix, so
most looks can be tuned without touching the matrix by hand.

## Film profiles

Each sim is a JSON file in `profiles/` — the filename stem is the profile key,
and `order` sets the swipe sequence. Both the Mac app and (later) the Pi load
the same files, so tuning a look on the Mac and copying the file to the camera
is the whole deployment story.

Processing runs entirely in float32 (0-1), quantising to 8-bit only at the
output, in this order: colour matrix → tone panel → curves → contrast →
HSL mixer → mono/saturation → colour grade → vignette → grain.

| Field | Meaning |
|---|---|
| `name`, `subtitle` | Shown on the rear display |
| `box_color` | RGB colour of the film-box UI |
| `color_matrix` | 3x3 RGB transform |
| `warmth`, `tint` | Red-blue / green-magenta shifts baked into the matrix |
| `exposure` | Stops, -2..+2 |
| `highlights`, `shadows`, `whites`, `blacks` | Lightroom-style tone panel, -1..+1 |
| `tone_curve` | Luminance curve as `[x, y]` control points (0-1), smoothly interpolated |
| `curve_red`, `curve_green`, `curve_blue` | Optional per-channel curves, same format |
| `contrast`, `saturation`, `brightness` | Global adjustments |
| `shadow_lift` | Lifts crushed blacks |
| `hsl` | 8-band mixer: `{"red": {"hue": deg, "sat": -1..1, "lum": -1..1}, ...}` (red, orange, yellow, green, aqua, blue, purple, magenta) |
| `grade` | Split toning: `{"shadows"/"midtones"/"highlights": {"hue": deg, "sat": 0..1}, "balance": -1..1}` |
| `monochrome` | Black & white sim (grade still applies — use it for sepia/selenium toning) |
| `grain_strength`, `grain_size` | 0 = none, ~0.3 = heavy; size 1 = fine, larger = coarser |
| `vignette`, `vignette_mid` | -1 dark corners .. +1 bright; midpoint of falloff |

## Tests

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```
