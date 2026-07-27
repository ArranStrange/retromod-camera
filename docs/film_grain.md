# Film Grain Engine

Research notes and design for the grain model in `processing/grain.py`.

## The physics (Boolean model)

The reference model for silver-halide grain (Newson, Faraj, Delon, Galerne —
*Realistic Film Grain Rendering*, IPOL 2017) treats the emulsion as a Boolean
model: opaque discs (developed grains) whose centres follow a Poisson process.
The grain density is tied to exposure so that expected coverage equals the
grey level `u`:

```
λ(u) = −log(1 − u) / (π · E[r²])
```

Two properties fall out of this and define what "real" grain looks like:

1. **Intensity response.** The fluctuation of a coverage process with mean `u`
   is `u(1 − u)`-shaped: zero at pure black (no grains) and pure white
   (fully covered), peaking in the midtones. Highlights and deep shadows are
   naturally clean — no masking hack required.
2. **Spatial correlation.** The disc radius sets a correlation length: grain
   is made of soft isotropic clumps, not independent per-pixel noise.

## The fast approximation we use

Monte Carlo simulation of the Boolean model is far too slow for a Pi Zero.
The follow-up work (*Analysis of a Physically Realistic Film Grain Model, and
a Gaussian Film Grain Synthesis Algorithm*, Newson et al. 2017) and the AV1
codec's film grain tool (Norkin & Birkbeck, DCC 2018) both show the accepted
shortcut: synthesise a **correlated Gaussian field** and scale it by an
**intensity response curve**.

`film_grain()` therefore:

1. Generates unit Gaussian white noise.
2. Blurs it to the target correlation length and renormalises to unit
   variance — this is the grain "clump" structure (AV1 uses an
   auto-regressive filter for the same purpose).
3. Applies it **multiplicatively** (density domain): `out = rgb · (1 + g)`
   with `g = noise · strength · (1 − u)^1.2`. The effective amplitude is
   therefore `u·(1 − u)^1.2` — the Boolean-model response shape, peaking
   below middle grey with strongly protected highlights. Because grain
   modulates exposure rather than being added as a grey layer, it blends
   with the image, preserves colour ratios, and cannot lift pure blacks.

## Parameter mapping (Fujifilm Grain Effect equivalents)

| Fuji control | Our field | Notes |
|---|---|---|
| Roughness weak/strong | `grain_strength` | ~0.15 weak, ~0.3 strong |
| Size small/large | `grain_size` | 1 = fine, 3+ = coarse clumps |

`grain_size` is defined **relative to the frame** (correlation length scales
with image dimensions), so grain looks identical on the 480p preview and a
full-resolution capture — the way grain has a fixed physical size on a
negative regardless of print size.

## Sources

- [Realistic Film Grain Rendering (IPOL 2017)](https://www.ipol.im/pub/art/2017/192/article_lr.pdf)
- [Analysis of a Physically Realistic Film Grain Model, and a Gaussian Film Grain Synthesis Algorithm](https://link.springer.com/chapter/10.1007/978-3-319-58771-4_16)
- [AV1 Film Grain Synthesis (Norkin & Birkbeck, DCC 2018)](https://norkin.org/pdf/DCC_2018_AV1_film_grain.pdf)
- [Fujifilm: Color Chrome and Film Grain Effects](https://www.fujifilm-x.com/en-us/stories/advanced-month-4-camera-features-13-color-chrome-and-film-grain-effects/)
- [Fstoppers: Fujifilm Grain Effect — How to Use It Right](https://fstoppers.com/education/fujifilms-grain-effect-more-useful-think-heres-how-actually-use-it-903376)
- [Alik Griffin: A Guide to Fujifilm's JPG Effects](https://alikgriffin.com/a-look-at-fujifilms-new-jpg-effects-clarity-color-chrome-grain/)
