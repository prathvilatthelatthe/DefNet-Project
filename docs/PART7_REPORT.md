# DeforestNet - Part 7 Implementation Report
## GradCAM Explainability

**Date:** 2026-03-30
**Status:** COMPLETED

---

## Summary

Part 7 implements GradCAM (Gradient-weighted Class Activation Mapping) for the U-Net model. This provides visual explanations showing WHERE the model looked and WHICH bands it relied on, building trust with forest officers and judges.

---

## What Was Implemented

### 1. GradCAM Core (`src/explainability/gradcam.py`)

**GradCAM Class:**
- Hooks into U-Net encoder layers to capture activations and gradients
- Generates per-class attention heatmaps
- Supports multiple target layers (bottleneck, encoder blocks)

**How It Works:**
```
Input Image → Forward Pass → Hook Activations
                                    ↓
Target Class Score → Backward Pass → Hook Gradients
                                    ↓
Weights = Global Avg Pool(Gradients)
Heatmap = ReLU(Σ Weights × Activations)
                                    ↓
Upsample to Input Size → Final Heatmap (256×256)
```

**BandImportanceAnalyzer Class:**
- Computes gradient-based importance for each of 11 input bands
- Shows which satellite bands contributed most to detection
- Supports batch analysis for stable statistics

**ExplainabilityReport Class:**
- Generates comprehensive reports combining all analyses
- Human-readable explanations with natural language
- Maps bands to real-world descriptions

### 2. Explanation Visualization (`src/explainability/explain_viz.py`)

| Function | Output |
|----------|--------|
| `heatmap_to_rgb()` | Colored heatmap (H, W, 3) |
| `overlay_heatmap()` | Heatmap overlaid on satellite image |
| `visualize_gradcam()` | Full figure: input + heatmap + overlay + band chart |
| `visualize_all_class_heatmaps()` | Grid of per-class heatmaps |
| `save_explanation_report()` | Complete report package |

### 3. Saved Output Files

Each explanation saves:
```
explanation_dir/
├── {name}_heatmap.npy         # Raw heatmap data
├── {name}_heatmap.png         # Colored heatmap image
├── {name}_overlay.png         # Heatmap overlaid on satellite
├── {name}_visualization.png   # Full visualization figure
├── {name}_report.json         # Structured report data
└── {name}_explanation.txt     # Human-readable explanation
```

---

## Sample Explanation Output

```
AI Classification: Logging (confidence: 94%)

The model detected tree removal patterns with exposed soil and reduced canopy.

Key factors in this decision:
  1. S2_B4_Red (30% importance): red light reflectance (soil exposure/brown areas)
  2. NDVI (25% importance): vegetation health index (green = high, brown = low)
  3. VV_VH_Ratio (18% importance): forest structure ratio (high = dense forest)

These top 3 bands account for 73% of the decision.
```

---

## Band Importance Descriptions

| Band | What AI Uses It For |
|------|---------------------|
| VV | Radar structural changes (tree trunks removed) |
| VH | Volume scattering changes (canopy gone) |
| B2 Blue | Water/mineral detection (mining pits) |
| B3 Green | Healthy vegetation indicator |
| B4 Red | Soil exposure / brown logging areas |
| B8 NIR | Vegetation vitality (dead vs alive) |
| NDVI | Overall vegetation health score |
| EVI | Forest density measure |
| SAVI | Soil-adjusted vegetation |
| VV/VH | Forest structure ratio |
| RVI | Radar canopy condition |

---

## Verification Results

```
============================================================
  VERIFICATION SUMMARY
============================================================
  GradCAM: [OK] PASS
  Band Importance: [OK] PASS
  Report Generation: [OK] PASS
  Visualization: [OK] PASS
============================================================

*** Part 7 GradCAM explainability is COMPLETE! ***
```

---

## Files Created

| File | Lines | Description |
|------|-------|-------------|
| `src/explainability/gradcam.py` | ~340 | GradCAM + Band Importance + Report |
| `src/explainability/explain_viz.py` | ~300 | Visualization utilities |
| `src/explainability/__init__.py` | ~30 | Package exports |
| `verify_part7.py` | ~170 | Verification script |

**Total new code:** ~840 lines

---

## Usage Examples

### Generate GradCAM Heatmap
```python
from src.explainability import GradCAM
from src.models import build_model

model = build_model()
gradcam = GradCAM(model, target_layer='bottleneck')

# For specific class
heatmap = gradcam.generate(image, target_class=1)  # Logging

# For all classes
all_heatmaps = gradcam.generate_all_classes(image)
```

### Band Importance Analysis
```python
from src.explainability import BandImportanceAnalyzer

analyzer = BandImportanceAnalyzer(model)
importance = analyzer.compute_band_importance(image)
# {'VV': 0.09, 'NDVI': 0.25, 'B4_Red': 0.30, ...}
```

### Full Explanation Report
```python
from src.explainability import ExplainabilityReport, save_explanation_report

reporter = ExplainabilityReport(model)
report = reporter.generate_report(image)
save_explanation_report(report, image, 'outputs/', name='sample_001')
```

---

## Next Part: Part 8 - Alert Generation System

### What Will Be Implemented:
- Deforestation detection logic (compare with baseline)
- Alert data structure creation
- Area calculation (pixels → hectares)
- SQLite alert storage
- Alert priority system

### Credentials Needed for Part 8:
**NONE** - Part 8 is purely local computation and SQLite database.

### Credentials Needed for Part 9 (Notifications):
- **Telegram Bot Token**: Create a bot via @BotFather on Telegram (FREE)
- **Gmail App Password**: For email alerts (FREE - Gmail account needed)

**Please create these before Part 9:**
1. Open Telegram → Search @BotFather → /newbot → Save the token
2. Gmail → Security → App Passwords → Create one for "DeforestNet"

---

**Part 7 Status: COMPLETE**

Ready to proceed with **Part 8: Alert Generation System**?
