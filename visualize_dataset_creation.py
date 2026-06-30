"""
DeforestNet - Dataset Creation Process Visualization
=====================================================
This script generates detailed visualizations explaining:
1. How synthetic satellite data is created step-by-step
2. How boundary masks are generated for each deforestation class
3. How spectral signatures differ across classes
4. Comparison: Synthetic vs Real satellite data format
5. The complete data pipeline from generation to model input

Output: outputs/visualizations/dataset_creation/ (multiple PNG figures)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.ndimage import zoom
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASS_NAMES, BAND_NAMES

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations" / "dataset_creation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color map for classes
CLASS_COLORS = {
    0: [0.13, 0.55, 0.13],  # Forest - Green
    1: [0.65, 0.33, 0.16],  # Logging - Brown
    2: [0.50, 0.00, 0.50],  # Mining - Purple
    3: [1.00, 0.84, 0.00],  # Agriculture - Gold
    4: [0.86, 0.08, 0.24],  # Fire - Crimson
    5: [0.50, 0.50, 0.50],  # Infrastructure - Gray
}

rng = np.random.RandomState(42)


def generate_noise(shape):
    """Multi-scale Perlin-like noise."""
    noise = np.zeros(shape, dtype=np.float32)
    for scale in [4, 8, 16, 32]:
        small = rng.rand(shape[0] // scale + 1, shape[1] // scale + 1)
        zoomed_arr = zoom(small, (shape[0] / small.shape[0], shape[1] / small.shape[1]), order=1)
        noise += zoomed_arr[:shape[0], :shape[1]] / scale
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


# ============================================================
# FIGURE 1: Step-by-Step Boundary Creation for Each Class
# ============================================================
def create_figure1_boundary_creation():
    """Show how each class boundary/mask is generated."""
    print("Creating Figure 1: Boundary Creation Process...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Step 1: How Deforestation Boundaries Are Created\n(Each class has a unique spatial pattern)",
                 fontsize=16, fontweight='bold', y=0.98)

    shape = (256, 256)

    # --- Forest: continuous canopy ---
    ax = axes[0, 0]
    mask = np.ones(shape, dtype=np.float32)
    noise = generate_noise(shape)
    mask = mask * (0.8 + 0.2 * noise)
    ax.imshow(mask, cmap='Greens', vmin=0, vmax=1)
    ax.set_title("Class 0: FOREST\n(Continuous canopy, natural texture noise)", fontsize=11, fontweight='bold')
    ax.text(128, 240, "Perlin noise → natural variation", ha='center', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))
    ax.axis('off')

    # --- Logging: irregular patches ---
    ax = axes[0, 1]
    mask = np.zeros(shape, dtype=np.float32)
    patches_info = []
    for _ in range(3):
        cx, cy = rng.randint(60, 196), rng.randint(60, 196)
        for _ in range(rng.randint(2, 4)):
            ox, oy = rng.randint(-25, 25), rng.randint(-25, 25)
            rx, ry = rng.randint(20, 50), rng.randint(20, 50)
            y, x = np.ogrid[:256, :256]
            ellipse = ((x - cx - ox) / rx) ** 2 + ((y - cy - oy) / ry) ** 2 <= 1
            mask[ellipse] = 1.0
            patches_info.append((cx + ox, cy + oy, rx, ry))
    ax.imshow(mask, cmap='YlOrBr', vmin=0, vmax=1)
    ax.set_title("Class 1: LOGGING\n(Overlapping ellipses → irregular patches)", fontsize=11, fontweight='bold')
    ax.text(128, 240, "Multiple random ellipses merged", ha='center', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='saddlebrown', alpha=0.8))
    ax.axis('off')

    # --- Mining: circular pits ---
    ax = axes[0, 2]
    mask = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[:256, :256]
    # Outer pit
    cx, cy, r = 128, 128, 55
    circle = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    mask[circle] = 1.0
    # Inner water
    inner = (x - cx) ** 2 + (y - cy) ** 2 <= (r * 0.6) ** 2
    mask[inner] = 0.7
    # Second smaller pit
    cx2, cy2, r2 = 190, 80, 30
    circle2 = (x - cx2) ** 2 + (y - cy2) ** 2 <= r2 ** 2
    mask[circle2] = 1.0
    ax.imshow(mask, cmap='Purples', vmin=0, vmax=1)
    # Draw annotation circles
    circle_outer = plt.Circle((128, 128), 55, fill=False, color='red', linewidth=2, linestyle='--')
    circle_water = plt.Circle((128, 128), 33, fill=False, color='cyan', linewidth=1.5, linestyle=':')
    ax.add_patch(circle_outer)
    ax.add_patch(circle_water)
    ax.set_title("Class 2: MINING\n(Circular pits with inner water bodies)", fontsize=11, fontweight='bold')
    ax.text(128, 240, "Outer pit (1.0) + inner water (0.7)", ha='center', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='purple', alpha=0.8))
    ax.axis('off')

    # --- Agriculture: grid fields ---
    ax = axes[1, 0]
    mask = np.zeros(shape, dtype=np.float32)
    field_size = 50
    for i in range(3):
        for j in range(3):
            if rng.rand() > 0.25:
                x1 = 25 + j * 55
                y1 = 25 + i * 55
                x2 = min(x1 + field_size, 256)
                y2 = min(y1 + field_size, 256)
                field = np.zeros((y2 - y1, x2 - x1))
                row_width = 5
                for row in range(0, y2 - y1, row_width * 2):
                    field[row:min(row + row_width, y2 - y1), :] = 1.0
                mask[y1:y2, x1:x2] = field
    ax.imshow(mask, cmap='YlGn', vmin=0, vmax=1)
    ax.set_title("Class 3: AGRICULTURE\n(Regular rectangular grid with crop rows)", fontsize=11, fontweight='bold')
    ax.text(128, 240, "Grid fields + alternating row pattern", ha='center', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='olive', alpha=0.8))
    ax.axis('off')

    # --- Fire: irregular spread ---
    ax = axes[1, 1]
    mask = np.zeros(shape, dtype=np.float32)
    cx, cy = 128, 128
    noise = generate_noise(shape)
    y, x = np.ogrid[:256, :256]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    base_radius = 60
    threshold = base_radius + 20 * (noise - 0.5)
    mask[dist < threshold] = 1.0
    # Fire fingers
    for _ in range(4):
        angle = rng.rand() * 2 * np.pi
        for t in range(35):
            px = int(cx + (base_radius + t) * np.cos(angle))
            py = int(cy + (base_radius + t) * np.sin(angle))
            if 0 <= px < 256 and 0 <= py < 256:
                w = 8
                mask[max(0, py - w):min(256, py + w), max(0, px - w):min(256, px + w)] = 1.0
    ax.imshow(mask, cmap='hot', vmin=0, vmax=1)
    ax.set_title("Class 4: FIRE / BURNT\n(Noise-based irregular spread + finger patterns)", fontsize=11, fontweight='bold')
    ax.text(128, 240, "Radial base + noise threshold + spread", ha='center', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8))
    ax.axis('off')

    # --- Infrastructure: roads + buildings ---
    ax = axes[1, 2]
    mask = np.zeros(shape, dtype=np.float32)
    # Horizontal road
    for x_pos in range(256):
        y_pos = int(128 + 15 * np.sin(x_pos * np.pi / 256))
        mask[max(0, y_pos - 4):min(256, y_pos + 4), x_pos] = 1.0
    # Vertical road
    for y_pos in range(256):
        x_pos = int(80 + 10 * np.sin(y_pos * np.pi / 256))
        mask[y_pos, max(0, x_pos - 3):min(256, x_pos + 3)] = 1.0
    # Buildings
    for _ in range(6):
        bx, by = rng.randint(20, 220), rng.randint(20, 220)
        bw, bh = rng.randint(12, 25), rng.randint(12, 25)
        mask[by:by + bh, bx:bx + bw] = 1.0
    ax.imshow(mask, cmap='Greys', vmin=0, vmax=1)
    ax.set_title("Class 5: INFRASTRUCTURE\n(Curved roads + rectangular buildings)", fontsize=11, fontweight='bold')
    ax.text(128, 240, "Sinusoidal lines + rectangles", ha='center', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='gray', alpha=0.8))
    ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUTPUT_DIR / "01_boundary_creation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved 01_boundary_creation.png")


# ============================================================
# FIGURE 2: Spectral Signature Application
# ============================================================
def create_figure2_spectral_signatures():
    """Show how spectral signatures are applied to create realistic bands."""
    print("Creating Figure 2: Spectral Signatures...")

    spectral_signatures = {
        "Forest":         {"VV": 0.60, "VH": 0.30, "B2": 0.10, "B3": 0.15, "B4": 0.12, "B8": 0.70},
        "Logging":        {"VV": 0.40, "VH": 0.35, "B2": 0.25, "B3": 0.30, "B4": 0.60, "B8": 0.25},
        "Mining":         {"VV": 0.30, "VH": 0.40, "B2": 0.50, "B3": 0.35, "B4": 0.30, "B8": 0.15},
        "Agriculture":    {"VV": 0.50, "VH": 0.25, "B2": 0.15, "B3": 0.25, "B4": 0.20, "B8": 0.55},
        "Fire":           {"VV": 0.25, "VH": 0.20, "B2": 0.08, "B3": 0.07, "B4": 0.10, "B8": 0.12},
        "Infrastructure": {"VV": 0.55, "VH": 0.45, "B2": 0.40, "B3": 0.40, "B4": 0.40, "B8": 0.35},
    }

    bands = ["VV", "VH", "B2", "B3", "B4", "B8"]
    class_names = list(spectral_signatures.keys())
    colors = ['#228B22', '#8B4513', '#800080', '#DAA520', '#DC143C', '#808080']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Step 2: Spectral Signatures — Each Class Has Unique Band Values\n"
                 "(Based on how real surfaces reflect light and radar)",
                 fontsize=14, fontweight='bold')

    # Bar chart
    ax = axes[0]
    x = np.arange(len(bands))
    width = 0.13
    for i, (cls, sigs) in enumerate(spectral_signatures.items()):
        values = [sigs[b] for b in bands]
        ax.bar(x + i * width, values, width, label=cls, color=colors[i], edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Spectral Band", fontsize=12)
    ax.set_ylabel("Reflectance / Backscatter (0–1)", fontsize=12)
    ax.set_title("Mean Spectral Values per Class", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(bands, fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', alpha=0.3)

    # Radar/spider chart
    ax = axes[1]
    angles = np.linspace(0, 2 * np.pi, len(bands), endpoint=False).tolist()
    angles += angles[:1]
    ax = fig.add_subplot(122, polar=True)
    for i, (cls, sigs) in enumerate(spectral_signatures.items()):
        values = [sigs[b] for b in bands] + [sigs[bands[0]]]
        ax.plot(angles, values, 'o-', linewidth=2, label=cls, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(bands, fontsize=11)
    ax.set_title("Spectral Profile (Radar Chart)", fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_spectral_signatures.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved 02_spectral_signatures.png")


# ============================================================
# FIGURE 3: Full Sample Generation Pipeline
# ============================================================
def create_figure3_sample_pipeline():
    """Show step-by-step: mask → spectral bands → derived indices → final 11-band image."""
    print("Creating Figure 3: Sample Generation Pipeline...")

    shape = (256, 256)

    # Create a multi-class mask
    label_mask = np.zeros(shape, dtype=np.int64)

    # Forest background
    forest_mask = np.ones(shape, dtype=np.float32) * (0.8 + 0.2 * generate_noise(shape))

    # Spectral signatures
    sigs = {
        0: {"VV": 0.60, "VH": 0.30, "B2": 0.10, "B3": 0.15, "B4": 0.12, "B8": 0.70},
        1: {"VV": 0.40, "VH": 0.35, "B2": 0.25, "B3": 0.30, "B4": 0.60, "B8": 0.25},
        2: {"VV": 0.30, "VH": 0.40, "B2": 0.50, "B3": 0.35, "B4": 0.30, "B8": 0.15},
        4: {"VV": 0.25, "VH": 0.20, "B2": 0.08, "B3": 0.07, "B4": 0.10, "B8": 0.12},
    }
    band_names_raw = ["VV", "VH", "B2", "B3", "B4", "B8"]

    # Build raw bands - start with forest
    raw_bands = np.zeros((6, 256, 256), dtype=np.float32)
    for i, bn in enumerate(band_names_raw):
        raw_bands[i] = sigs[0][bn] * forest_mask + rng.randn(256, 256) * 0.02

    # Add logging patch
    log_mask = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[:256, :256]
    ellipse = ((x - 80) / 40) ** 2 + ((y - 100) / 35) ** 2 <= 1
    log_mask[ellipse] = 1.0
    label_mask[ellipse] = 1
    for i, bn in enumerate(band_names_raw):
        raw_bands[i][ellipse] = sigs[1][bn] + rng.randn(np.sum(ellipse)) * 0.03

    # Add mining pit
    mine_mask = np.zeros(shape, dtype=np.float32)
    circle = (x - 190) ** 2 + (y - 160) ** 2 <= 40 ** 2
    mine_mask[circle] = 1.0
    label_mask[circle] = 2
    for i, bn in enumerate(band_names_raw):
        raw_bands[i][circle] = sigs[2][bn] + rng.randn(np.sum(circle)) * 0.03

    # Add fire scar
    fire_mask = np.zeros(shape, dtype=np.float32)
    noise = generate_noise(shape)
    dist = np.sqrt((x - 140) ** 2 + (y - 60) ** 2)
    fire_region = dist < (35 + 15 * (noise - 0.5))
    fire_mask[fire_region] = 1.0
    label_mask[fire_region] = 4
    for i, bn in enumerate(band_names_raw):
        raw_bands[i][fire_region] = sigs[4][bn] + rng.randn(np.sum(fire_region)) * 0.02

    raw_bands = np.clip(raw_bands, 0, 1)

    # Compute derived indices
    VV, VH, B2, B3, B4, B8 = raw_bands
    eps = 1e-8
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    evi = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + eps)
    savi = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)
    vv_vh = np.clip(VV / (VH + eps) / 20, 0, 1)
    rvi = VV / (VV + VH + eps)

    derived = [ndvi, evi, savi, vv_vh, rvi]
    derived_names = ["NDVI", "EVI", "SAVI", "VV/VH Ratio", "RVI"]

    # --- Plot ---
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("Step 3: Complete Sample Generation Pipeline\n"
                 "Boundary Mask → Apply Spectral Values → Compute Indices → Final 11-Band Image",
                 fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(3, 6, hspace=0.4, wspace=0.3)

    # Row 1: Class masks + combined label mask
    mask_data = [
        (forest_mask, "Forest Background", 'Greens'),
        (log_mask, "Logging Patch", 'YlOrBr'),
        (mine_mask, "Mining Pit", 'Purples'),
        (fire_mask, "Fire Scar", 'hot'),
    ]
    for i, (m, title, cmap) in enumerate(mask_data):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    # Combined label mask
    ax = fig.add_subplot(gs[0, 4:6])
    color_mask = np.zeros((256, 256, 3))
    for cls_idx, color in CLASS_COLORS.items():
        color_mask[label_mask == cls_idx] = color
    ax.imshow(color_mask)
    ax.set_title("Combined Ground Truth Mask\n(This is what model learns to predict)", fontsize=11, fontweight='bold')
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(6) if i in [0, 1, 2, 4]]
    ax.legend(handles=patches, fontsize=8, loc='lower right')
    ax.axis('off')

    # Row 2: Raw spectral bands
    cmaps_band = ['gray', 'gray', 'Blues', 'Greens', 'Reds', 'RdYlGn']
    for i in range(6):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(raw_bands[i], cmap=cmaps_band[i], vmin=0, vmax=1)
        ax.set_title(f"Band {i + 1}: {band_names_raw[i]}\n({'SAR Radar' if i < 2 else 'Optical'})",
                     fontsize=9, fontweight='bold')
        ax.axis('off')

    # Row 3: Derived indices + final composite
    idx_cmaps = ['RdYlGn', 'RdYlGn', 'RdYlGn', 'coolwarm', 'coolwarm']
    for i in range(5):
        ax = fig.add_subplot(gs[2, i])
        d = derived[i]
        ax.imshow(d, cmap=idx_cmaps[i])
        ax.set_title(f"Index {i + 1}: {derived_names[i]}\n(Computed from raw bands)",
                     fontsize=9, fontweight='bold')
        ax.axis('off')

    # Final: False-color composite
    ax = fig.add_subplot(gs[2, 5])
    rgb = np.stack([
        np.clip(raw_bands[4], 0, 1),  # B4 → Red
        np.clip(raw_bands[5] * 0.5, 0, 1),  # B8 → Green (scaled)
        np.clip(raw_bands[2], 0, 1),  # B2 → Blue
    ], axis=-1)
    ax.imshow(rgb)
    ax.set_title("False-Color Composite\n(B4=R, B8=G, B2=B)", fontsize=10, fontweight='bold')
    ax.axis('off')

    # Add arrows / annotations
    fig.text(0.08, 0.68, "STEP 1\nBoundary\nMasks", fontsize=12, fontweight='bold', color='#333',
             ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0f0e0'))
    fig.text(0.08, 0.40, "STEP 2\nSpectral\nBands", fontsize=12, fontweight='bold', color='#333',
             ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0e0f0'))
    fig.text(0.08, 0.12, "STEP 3\nDerived\nIndices", fontsize=12, fontweight='bold', color='#333',
             ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0e0e0'))

    plt.savefig(OUTPUT_DIR / "03_sample_pipeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved 03_sample_pipeline.png")


# ============================================================
# FIGURE 4: Synthetic vs Real Data Comparison
# ============================================================
def create_figure4_synthetic_vs_real():
    """Explain differences between synthetic and real satellite data."""
    print("Creating Figure 4: Synthetic vs Real Comparison...")

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle("Synthetic Data vs Real Satellite Data — What's Different and What's the Same",
                 fontsize=15, fontweight='bold')

    shape = (256, 256)

    # === TOP ROW: SYNTHETIC ===
    # Synthetic NDVI
    ax = axes[0, 0]
    forest_bg = 0.7 + 0.1 * generate_noise(shape)
    synth_ndvi = forest_bg.copy()
    y, x = np.ogrid[:256, :256]
    # Clean logging patch
    ellipse = ((x - 100) / 45) ** 2 + ((y - 130) / 40) ** 2 <= 1
    synth_ndvi[ellipse] = 0.15 + rng.randn(np.sum(ellipse)) * 0.03
    ax.imshow(synth_ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
    ax.set_title("Synthetic NDVI\n(Clean boundaries, uniform values)", fontsize=10, fontweight='bold')
    ax.text(100, 130, "Logging", ha='center', fontsize=9, color='white', fontweight='bold')
    ax.axis('off')

    # Synthetic mask
    ax = axes[0, 1]
    synth_mask = np.zeros(shape, dtype=np.int64)
    synth_mask[ellipse] = 1
    color_mask = np.zeros((256, 256, 3))
    color_mask[synth_mask == 0] = CLASS_COLORS[0]
    color_mask[synth_mask == 1] = CLASS_COLORS[1]
    ax.imshow(color_mask)
    ax.set_title("Synthetic Ground Truth\n(Perfect pixel-level labels)", fontsize=10, fontweight='bold')
    ax.axis('off')

    # Synthetic band histogram
    ax = axes[0, 2]
    forest_vals = synth_ndvi[synth_mask == 0].flatten()
    logging_vals = synth_ndvi[synth_mask == 1].flatten()
    ax.hist(forest_vals, bins=50, alpha=0.7, color='green', label='Forest', density=True)
    ax.hist(logging_vals, bins=50, alpha=0.7, color='brown', label='Logging', density=True)
    ax.set_title("Synthetic NDVI Distribution\n(Well-separated, little overlap)", fontsize=10, fontweight='bold')
    ax.legend()
    ax.set_xlabel("NDVI")

    # Synthetic properties box
    ax = axes[0, 3]
    ax.axis('off')
    props_synth = [
        "✓ SYNTHETIC DATA",
        "",
        "Boundaries: Clean geometric shapes",
        "   (ellipses, circles, rectangles)",
        "",
        "Noise: Controlled Perlin noise",
        "   (smooth, predictable)",
        "",
        "Labels: Automatically generated",
        "   (100% accurate, zero effort)",
        "",
        "Spectral values: From lookup table",
        "   (mean ± std per class)",
        "",
        "Clouds: None (always clear)",
        "",
        "Time to create: ~2 minutes",
        "   for 1200 labeled images",
    ]
    ax.text(0.05, 0.95, '\n'.join(props_synth), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.9))

    # === BOTTOM ROW: SIMULATED "REAL" ===
    # Real-like NDVI (messy boundaries, cloud shadows, mixed pixels)
    ax = axes[1, 0]
    real_ndvi = 0.65 + 0.15 * generate_noise(shape)  # noisier forest
    # Add cloud shadow
    cloud_y, cloud_x = 40, 160
    for cy in range(cloud_y, cloud_y + 60):
        for cx in range(cloud_x, min(256, cloud_x + 80)):
            if 0 <= cy < 256:
                real_ndvi[cy, cx] *= 0.4 + rng.rand() * 0.2
    # Messy deforestation with fuzzy edges
    dist_field = np.sqrt(((x - 100) / 50) ** 2 + ((y - 140) / 42) ** 2)
    fuzzy_edge = 1.0 / (1.0 + np.exp(10 * (dist_field - 1.0)))  # sigmoid boundary
    deforest_noise = rng.randn(256, 256) * 0.08
    real_logging_mask = fuzzy_edge > 0.5 + deforest_noise
    real_ndvi[real_logging_mask] = 0.12 + rng.randn(np.sum(real_logging_mask)) * 0.08  # more noise
    # Scattered pixels (mixed)
    scatter = rng.rand(256, 256) < 0.02
    real_ndvi[scatter] = rng.rand(np.sum(scatter)) * 0.5
    ax.imshow(real_ndvi, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
    ax.set_title("Real-World NDVI (simulated)\n(Fuzzy edges, cloud shadow, noise)", fontsize=10, fontweight='bold')
    ax.text(160, 60, "Cloud\nshadow", ha='center', fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
    ax.text(100, 140, "Logging\n(fuzzy edge)", ha='center', fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='brown', alpha=0.7))
    ax.axis('off')

    # Real-like mask (imperfect labels)
    ax = axes[1, 1]
    real_mask = np.zeros(shape, dtype=np.int64)
    # Manually drawn (polygon-style, not pixel-perfect)
    real_mask[real_logging_mask] = 1
    # Some mislabeled pixels at edges
    edge_noise = rng.rand(256, 256) < 0.05
    boundary = np.abs(dist_field - 1.0) < 0.15
    real_mask[boundary & edge_noise] = 0  # some boundary pixels wrong
    color_mask_real = np.zeros((256, 256, 3))
    color_mask_real[real_mask == 0] = CLASS_COLORS[0]
    color_mask_real[real_mask == 1] = CLASS_COLORS[1]
    ax.imshow(color_mask_real)
    ax.set_title("Real Ground Truth (simulated)\n(Manual annotation, imperfect edges)", fontsize=10, fontweight='bold')
    ax.axis('off')

    # Real band histogram
    ax = axes[1, 2]
    real_forest = real_ndvi[real_mask == 0].flatten()
    real_logging = real_ndvi[real_mask == 1].flatten()
    ax.hist(real_forest, bins=50, alpha=0.7, color='green', label='Forest', density=True)
    ax.hist(real_logging, bins=50, alpha=0.7, color='brown', label='Logging', density=True)
    ax.set_title("Real NDVI Distribution\n(More overlap, harder to classify)", fontsize=10, fontweight='bold')
    ax.legend()
    ax.set_xlabel("NDVI")

    # Real properties box
    ax = axes[1, 3]
    ax.axis('off')
    props_real = [
        "✗ REAL SATELLITE DATA",
        "",
        "Boundaries: Irregular, fuzzy edges",
        "   (no clean separation)",
        "",
        "Noise: Sensor noise + atmosphere",
        "   (speckle, haze, cloud shadow)",
        "",
        "Labels: Manual annotation needed",
        "   (200+ hours for 1000 images)",
        "",
        "Spectral values: Measured by sensor",
        "   (variable with season/weather)",
        "",
        "Clouds: 30-70% of images affected",
        "",
        "Time to create: 3-6 months",
        "   (data collection + labeling)",
    ]
    ax.text(0.05, 0.95, '\n'.join(props_real), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUTPUT_DIR / "04_synthetic_vs_real.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved 04_synthetic_vs_real.png")


# ============================================================
# FIGURE 5: Real Dataset Samples from our .npy files
# ============================================================
def create_figure5_actual_dataset_samples():
    """Load and visualize actual samples from our generated dataset."""
    print("Creating Figure 5: Actual Dataset Samples...")

    data_dir = PROJECT_ROOT / "data" / "synthetic"
    images = np.load(data_dir / "test_images.npy")
    masks = np.load(data_dir / "test_masks.npy")

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(f"Actual Generated Dataset — 4 Test Samples from test_images.npy\n"
                 f"Dataset: {images.shape[0]} images, {images.shape[1]} bands, "
                 f"{images.shape[2]}×{images.shape[3]} pixels",
                 fontsize=15, fontweight='bold')

    for sample_idx in range(4):
        img = images[sample_idx * 15]  # Pick spread-out samples
        mask = masks[sample_idx * 15]

        # Row per sample: VV | VH | B4(Red) | B8(NIR) | NDVI | False Color | Ground Truth
        bands_to_show = [
            (0, "VV (SAR)", 'gray'),
            (1, "VH (SAR)", 'gray'),
            (4, "B4 (Red)", 'Reds'),
            (5, "B8 (NIR)", 'RdYlGn'),
            (6, "NDVI", 'RdYlGn'),
        ]

        for col, (band_idx, title, cmap) in enumerate(bands_to_show):
            ax = fig.add_subplot(4, 7, sample_idx * 7 + col + 1)
            ax.imshow(img[band_idx], cmap=cmap)
            if sample_idx == 0:
                ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(f"Sample {sample_idx + 1}", fontsize=11, fontweight='bold')

        # False color
        ax = fig.add_subplot(4, 7, sample_idx * 7 + 6)
        rgb = np.stack([
            np.clip(img[4], 0, 1),
            np.clip(img[5] * 0.5, 0, 1),
            np.clip(img[2], 0, 1),
        ], axis=-1)
        ax.imshow(rgb)
        if sample_idx == 0:
            ax.set_title("False Color", fontsize=10, fontweight='bold')
        ax.axis('off')

        # Ground truth mask
        ax = fig.add_subplot(4, 7, sample_idx * 7 + 7)
        color_mask = np.zeros((256, 256, 3))
        for cls_idx, color in CLASS_COLORS.items():
            color_mask[mask == cls_idx] = color
        ax.imshow(color_mask)
        if sample_idx == 0:
            ax.set_title("Ground Truth", fontsize=10, fontweight='bold')
        ax.axis('off')

        # Class distribution for this sample
        unique, counts = np.unique(mask, return_counts=True)
        class_info = ", ".join([f"{CLASS_NAMES[u]}:{counts[i] / mask.size * 100:.0f}%"
                                for i, u in enumerate(unique)])

    # Legend at bottom
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=f"{i}: {CLASS_NAMES[i]}") for i in range(6)]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=11, frameon=True,
               fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(OUTPUT_DIR / "05_actual_dataset_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved 05_actual_dataset_samples.png")


# ============================================================
# FIGURE 6: End-to-End Data Pipeline Summary
# ============================================================
def create_figure6_pipeline_summary():
    """Create a visual summary of the entire pipeline."""
    print("Creating Figure 6: Pipeline Summary Diagram...")

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')
    ax.set_title("Complete Data Pipeline: From Generation to Prediction",
                 fontsize=18, fontweight='bold', pad=20)

    def draw_box(x, y, w, h, text, color, fontsize=9, textcolor='white'):
        box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                       facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h / 2 + (len(lines) / 2 - i - 0.5) * 1.2,
                    line, ha='center', va='center', fontsize=fontsize,
                    fontweight='bold' if i == 0 else 'normal', color=textcolor)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color='#333', lw=2))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.8, label, ha='center', fontsize=8, color='#555', style='italic')

    # Row 1: Data Generation
    draw_box(2, 48, 18, 8, "1. BOUNDARY GENERATION\n\nGenerate spatial masks\n(ellipses, circles, grids)\nfor each class", '#2E7D32')
    draw_arrow(20, 52, 24, 52, "masks")
    draw_box(24, 48, 18, 8, "2. SPECTRAL MAPPING\n\nApply band values\nVV,VH,B2,B3,B4,B8\nfrom lookup table", '#1565C0')
    draw_arrow(42, 52, 46, 52, "6 bands")
    draw_box(46, 48, 18, 8, "3. INDEX COMPUTATION\n\nNDVI, EVI, SAVI\nVV/VH Ratio, RVI\n→ 5 more channels", '#6A1B9A')
    draw_arrow(64, 52, 68, 52, "11 bands")
    draw_box(68, 48, 18, 8, "4. SAVE DATASET\n\ntrain: 800 images\nval: 200 images\ntest: 200 images\n(.npy format)", '#E65100')

    # Row 2: Training
    draw_box(2, 32, 18, 8, "5. DATA LOADING\n\nDataLoader reads\n.npy → Tensor\nAugmentation applied", '#00695C')
    draw_arrow(20, 36, 24, 36, "batches")
    draw_box(24, 32, 18, 8, "6. U-NET MODEL\n\nEncoder: 11→1024 ch\nBottleneck: 16×16\nDecoder: 1024→6 ch", '#C62828')
    draw_arrow(42, 36, 46, 36, "prediction")
    draw_box(46, 32, 18, 8, "7. LOSS & OPTIMIZE\n\nCrossEntropy + Dice\nAdam optimizer\nlr = 0.001", '#AD1457')
    draw_arrow(64, 36, 68, 36, "best.pt")
    draw_box(68, 32, 18, 8, "8. SAVED MODEL\n\nbest.pt (epoch 6)\nAccuracy: 99.7%\nMean IoU: 98.4%", '#F57F17')

    # Row 3: Inference
    draw_box(2, 16, 18, 8, "9. NEW IMAGE\n\nSatellite captures\n11-band image\n(or test sample)", '#37474F')
    draw_arrow(20, 20, 24, 20, "input")
    draw_box(24, 16, 18, 8, "10. INFERENCE\n\nLoad best.pt\nForward pass\n→ 6-class mask", '#BF360C')
    draw_arrow(42, 20, 46, 20, "mask")
    draw_box(46, 16, 18, 8, "11. ANALYSIS\n\nCount pixels per class\nCalculate area (ha)\nDetermine severity", '#1A237E', textcolor='white')
    draw_arrow(64, 20, 68, 20, "alert")
    draw_box(68, 16, 18, 8, "12. ALERT & NOTIFY\n\nCreate alert in DB\nSend Telegram msg\nSend Email to officer", '#1B5E20')

    # Arrows connecting rows
    draw_arrow(77, 48, 77, 40.5, "")
    draw_arrow(11, 40, 11, 24.5, "")
    ax.annotate("saved dataset", xy=(77, 44), fontsize=8, color='#555', ha='center', style='italic')
    ax.annotate("trained model", xy=(15, 32), fontsize=8, color='#555', ha='left', style='italic')

    # Bottom note
    ax.text(50, 6, "KEY INSIGHT: Steps 1-4 use synthetic data. Steps 5-12 are production-ready.\n"
                    "To deploy with real data: Replace Step 1-4 with Copernicus API download + manual labeling.\n"
                    "All other steps remain EXACTLY the same — zero code changes needed.",
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2))

    plt.savefig(OUTPUT_DIR / "06_pipeline_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved 06_pipeline_summary.png")


# ============================================================
# FIGURE 7: Band-by-band explanation
# ============================================================
def create_figure7_band_explanation():
    """Explain what each of the 11 bands represents."""
    print("Creating Figure 7: Band Explanation...")

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.axis('off')
    ax.set_title("The 11 Input Bands — What Each One Measures",
                 fontsize=16, fontweight='bold', pad=20)

    bands_info = [
        ("VV", "Vertical-Vertical", "SAR Radar", "Sentinel-1",
         "Measures surface roughness. Forest = high (rough canopy), Water = low (smooth).",
         "Works through clouds, rain, darkness. Key for monsoon monitoring.", "#455A64"),
        ("VH", "Vertical-Horizontal", "SAR Radar", "Sentinel-1",
         "Cross-polarization. Sensitive to volume scattering in vegetation.",
         "Higher for dense forest, lower for bare soil/water.", "#546E7A"),
        ("B2", "Blue (490 nm)", "Optical", "Sentinel-2",
         "Detects water bodies (high reflectance) and soil type.",
         "Used in EVI computation. Blocked by clouds.", "#1565C0"),
        ("B3", "Green (560 nm)", "Optical", "Sentinel-2",
         "Vegetation reflects green light (why forests look green).",
         "Higher for healthy vegetation.", "#2E7D32"),
        ("B4", "Red (665 nm)", "Optical", "Sentinel-2",
         "Vegetation absorbs red light. Low = healthy, High = bare/burnt.",
         "Key component of NDVI. Logging shows high B4.", "#C62828"),
        ("B8", "NIR (842 nm)", "Optical", "Sentinel-2",
         "Near-infrared. Healthy vegetation strongly reflects NIR.",
         "Most important optical band for forest monitoring.", "#4E342E"),
        ("NDVI", "(B8-B4)/(B8+B4)", "Computed", "Derived",
         "Normalized Difference Vegetation Index. Range: -1 to +1.",
         "Forest≈0.7, Logging≈0.15, Water≈-0.1. Primary indicator.", "#388E3C"),
        ("EVI", "Enhanced Veg Index", "Computed", "Derived",
         "Improved NDVI. Better for dense forest (reduces saturation).",
         "More accurate than NDVI in tropical forests.", "#558B2F"),
        ("SAVI", "Soil-Adjusted VI", "Computed", "Derived",
         "Like NDVI but corrects for bare soil influence.",
         "Better in areas with sparse vegetation or mixed pixels.", "#827717"),
        ("VV/VH", "Ratio", "Computed", "Derived",
         "Radar polarization ratio. Indicates vegetation structure.",
         "Forest has high ratio (complex structure), bare soil = low.", "#4527A0"),
        ("RVI", "Radar Veg Index", "Computed", "Derived",
         "VV/(VV+VH). Cloud-independent vegetation measure.",
         "Crucial during monsoon when optical bands are unusable.", "#6A1B9A"),
    ]

    y_start = 0.92
    for i, (name, full_name, btype, source, desc, note, color) in enumerate(bands_info):
        y = y_start - i * 0.083

        # Band number
        ax.text(0.01, y, f"Band {i + 1}", fontsize=9, fontweight='bold', color='white',
                transform=ax.transAxes, va='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

        # Name
        ax.text(0.08, y, f"{name}", fontsize=11, fontweight='bold',
                transform=ax.transAxes, va='center', color=color)

        # Full name + type
        ax.text(0.15, y, f"({full_name})  [{btype} — {source}]", fontsize=9,
                transform=ax.transAxes, va='center', color='#555')

        # Description
        ax.text(0.50, y, desc, fontsize=9, transform=ax.transAxes, va='center')

        # Note
        ax.text(0.50, y - 0.025, note, fontsize=8, transform=ax.transAxes,
                va='center', color='#777', style='italic')

        # Separator line
        if i < 10:
            ax.plot([0.01, 0.99], [y - 0.045, y - 0.045], color='#eee',
                    linewidth=0.5, transform=ax.transAxes, clip_on=False)

    # Category labels
    ax.text(0.98, 0.88, "SAR\n(works in\nmonsoon)", fontsize=10, fontweight='bold',
            transform=ax.transAxes, ha='right', va='center', color='white',
            bbox=dict(boxstyle='round', facecolor='#37474F'))
    ax.text(0.98, 0.62, "OPTICAL\n(needs clear\nweather)", fontsize=10, fontweight='bold',
            transform=ax.transAxes, ha='right', va='center', color='white',
            bbox=dict(boxstyle='round', facecolor='#1565C0'))
    ax.text(0.98, 0.28, "COMPUTED\n(derived from\nabove bands)", fontsize=10, fontweight='bold',
            transform=ax.transAxes, ha='right', va='center', color='white',
            bbox=dict(boxstyle='round', facecolor='#4A148C'))

    plt.savefig(OUTPUT_DIR / "07_band_explanation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved 07_band_explanation.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DeforestNet - Dataset Creation Visualization")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")
    print()

    create_figure1_boundary_creation()
    create_figure2_spectral_signatures()
    create_figure3_sample_pipeline()
    create_figure4_synthetic_vs_real()
    create_figure5_actual_dataset_samples()
    create_figure6_pipeline_summary()
    create_figure7_band_explanation()

    print()
    print("=" * 60)
    print(f"  All 7 figures saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
    print()
    print("Use these figures to explain:")
    print("  Fig 1 → How boundary shapes are created for each class")
    print("  Fig 2 → How spectral signatures make each class unique")
    print("  Fig 3 → Complete pipeline: mask → bands → indices → 11-channel image")
    print("  Fig 4 → Why synthetic vs real, and what's different")
    print("  Fig 5 → Actual samples from your generated dataset")
    print("  Fig 6 → End-to-end pipeline diagram (generation → training → inference → alert)")
    print("  Fig 7 → What each of the 11 bands measures and why")
