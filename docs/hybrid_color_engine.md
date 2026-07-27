# Hybrid Color Science Engine: Technical Specification & Python Implementation

A hybrid Image Signal Processing (ISP) pipeline combining **Fujifilm's emotional "Memory Color" architecture** with **Leica's optical dimensionality and micro-contrast**.

---

## Architecture Overview

```
                                  [ RAW / Linear RGB Input ]
                                               │
                                               ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │ 1. LEICA FRONT-END                                               │
             │    • Highlight Protection (-0.35 EV Baseline Offset)             │
             │    • Warm Auto White Balance Bias (+150K to +200K)              │
             └──────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │ 2. HYBRID TONAL CURVE                                            │
             │    • Linear-to-Log Space Mapping                                 │
             │    • Fuji Filmic Sigmoidal S-Curve Compression Knee              │
             │    • Leica Steep Shadow Density Slope                            │
             └──────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │ 3. LEICA SPATIAL MICRO-CONTRAST ENGINE                           │
             │    • Rec. 709 Midtone Luminance Isolation Mask                   │
             │    • High-Frequency Spatial Detail Boosting (Unsharp Mask)       │
             └──────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │ 4. FUJIFILM PROCEDURAL GRAIN ENGINE                              │
             │    • Luminance-Weighted Gaussian Noise Injection                 │
             │    • High-Frequency Masking to Protect Highlights / Deep Shadows │
             └──────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                                 [ Processed Output Frame ]
```

---

## Stage Breakdown

| Stage | Design Philosophy | Technical Implementation |
| :--- | :--- | :--- |
| **1. Exposure Guard** | Leica Highlight Safety | Applies a default $-0.35	ext{ EV}$ exposure offset to prevent digital sensor clipping and preserve highlight detail. |
| **2. White Balance** | Leica Warmth & Depth | Applies a warm white balance gain ($+150	ext{K} 	ext{ to } +200	ext{K}$ equivalent offset) while preserving skin tone vector fidelity. |
| **3. Tone Curve** | Fuji Compression / Leica Shadows | Converts linear scene-referred data into logarithmic space, applying a sigmoidal knee to compress highlights and a steep slope for dense shadows. |
| **4. Micro-Contrast** | Leica 3D "Pop" | Extracts high-frequency spatial detail via Gaussian blur and selectively amplifies midtones using a Gaussian-weighted luminance mask. |
| **5. Organic Grain** | Fuji Analogue Texture | Synthesizes pseudo-random Gaussian noise modulated by the midtone luminance mask to emulate real silver-halide film structure. |

---

## Vectorized Python Core Engine (`hybrid_isp_engine.py`)

The following implementation uses **NumPy** and **OpenCV** for vectorized processing. All mathematical operations are executed on contiguous C-level memory buffers to process multi-megapixel images efficiently.

```python
import cv2
import numpy as np

class HybridISPEngine:
    """
    A hybrid ISP Engine combining Leica's micro-contrast & exposure strategy
    with Fujifilm's color rendition and highlight roll-off curves.
    """
    def __init__(self, 
                 exposure_offset=-0.35,       # Leica highlight safety offset (-0.35 EV)
                 wb_gains=(1.05, 1.00, 0.94),  # Warm AWB bias (R, G, B)
                 micro_contrast=0.18,         # Leica 3D Pop (Unsharp Mask Gain)
                 grain_amount=0.04):          # Fuji Midtone Grain intensity
        
        self.exposure_offset = exposure_offset
        self.wb_gains = np.array(wb_gains, dtype=np.float32)
        self.micro_contrast = micro_contrast
        self.grain_amount = grain_amount

    def _linear_to_log(self, x: np.ndarray) -> np.ndarray:
        """Convert linear scene-referred RGB data to a normalized logarithmic workspace."""
        x = np.maximum(0.0, x)
        return np.where(x > 0.01, 0.2 * np.log(x + 0.001) + 0.8, x * 10.0)

    def _log_to_linear(self, x: np.ndarray) -> np.ndarray:
        """Convert logarithmic workspace back to linear RGB space."""
        return np.where(x > 0.1, np.exp((x - 0.8) / 0.2) - 0.001, x / 10.0)

    def _apply_filmic_curve(self, log_rgb: np.ndarray) -> np.ndarray:
        """Fuji-style Sigmoidal S-Curve applied in Log Space."""
        v = np.clip(log_rgb, 0.0, 1.0)
        return (v * (1.0 + 0.5 * v)) / (v**2 + 0.4 * v + 0.6)

    def process_frame(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Processes an input RGB float32 image [0.0, 1.0].
        
        Parameters:
            img_rgb (np.ndarray): Normalized floating point image array (H, W, 3).
            
        Returns:
            np.ndarray: Processed image array clamped to [0.0, 1.0].
        """
        frame = img_rgb.astype(np.float32)

        # -------------------------------------------------------------
        # PASS 1: Leica Front-End (Highlight Protection + Warm AWB)
        # -------------------------------------------------------------
        scale = 2.0 ** self.exposure_offset
        frame = frame * scale * self.wb_gains

        # -------------------------------------------------------------
        # PASS 2: Hybrid Tonal Curve (Log Workspace + Sigmoidal Knee)
        # -------------------------------------------------------------
        log_frame = self._linear_to_log(frame)
        curved_log = self._apply_filmic_curve(log_frame)
        frame = self._log_to_linear(curved_log)

        # -------------------------------------------------------------
        # PASS 3: Leica Midtone Micro-Contrast Engine
        # -------------------------------------------------------------
        # Rec. 709 Luminance extraction
        luminance = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]

        # Midtone Mask: Peaks at midtone brightness (~0.5)
        midtone_mask = np.exp(-((luminance - 0.5) ** 2) / 0.08)
        midtone_mask_3ch = np.dstack([midtone_mask] * 3)

        # Extract high-frequency spatial detail via Gaussian blur
        blurred = cv2.GaussianBlur(frame, (9, 9), sigmaX=2.0)
        high_freq_detail = frame - blurred

        # Inject micro-contrast strictly into midtones
        frame += high_freq_detail * self.micro_contrast * midtone_mask_3ch

        # -------------------------------------------------------------
        # PASS 4: Fuji Procedural Midtone Grain Engine
        # -------------------------------------------------------------
        h, w, _ = frame.shape
        noise = np.random.normal(0.0, self.grain_amount, (h, w, 1)).astype(np.float32)
        frame += noise * midtone_mask_3ch

        # Final clip to valid display range
        return np.clip(frame, 0.0, 1.0)


# =====================================================================
# Pipeline Runner / Example Usage
# =====================================================================
if __name__ == "__main__":
    # Initialize engine with target parameters
    engine = HybridISPEngine(
        exposure_offset=-0.35, 
        micro_contrast=0.20, 
        grain_amount=0.03
    )

    # Load image (OpenCV loads BGR; convert to RGB float32)
    input_bgr = cv2.imread("test_input.jpg")
    
    if input_bgr is None:
        print("No test image found. Generating test gradient canvas...")
        h, w = 1080, 1920
        grid = np.linspace(0, 1, w, dtype=np.float32)
        input_rgb = np.tile(grid, (h, 1, 3))
    else:
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Execute processing pipeline
    output_rgb = engine.process_frame(input_rgb)

    # Convert back to 8-bit BGR for saving
    output_bgr = cv2.cvtColor((output_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("hybrid_processed_output.jpg", output_bgr)
    print("Processing complete. Saved output to 'hybrid_processed_output.jpg'.")
